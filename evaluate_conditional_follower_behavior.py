"""
Finite-State follower – slower/longer sim + trajectory & velocity plots
────────────────────────────────────────────────────────────────────────
• Leader: 0.05 m/s move, 0.05 m/s descent
• Phase length: 10 s  (200 steps @ 0.05 s)
• Extra 20-s buffer after land
• Generates two matplotlib figures:
    1) XY path of leader & follower
    2) vx, vy, vz curves for both agents
• prints: classification accuracy, trajectory RMSE, action-variance
"""

# ─── Imports ─────────────────────────────────────────────────────
import os, time
from functools import partial
import numpy as np
import jax, jax.numpy as jnp, jax_dataclasses as jdc
from jax import jit
from flax import linen as nn
from orbax import checkpoint as oc
import matplotlib.pyplot as plt
import pybullet as p, pybullet_data

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State
from crazy_rl.utils.jax_spaces import Box, Space

# ─── 1. Global constants ────────────────────────────────────────
SIM_DT              = 0.05   # 20 Hz physics
STEPS_PER_PHASE     = 200    # 10 s × 3 ≈ 30 s
TAIL_STEPS          = 400    # +20 s buffer
ENV_SIZE            = 2.0
MAX_ACT_VEL         = 1.0

LEADER_MOVE_SPEED   = 0.05
LEADER_DESCENT_RATE = 0.05

R_HOVER      = 0.5
FOLLOW_DIST  = 0.25
MIN_DIST     = 0.3
KP_DEFAULT   = 2.0
KP_LAND      = 1.2
ASCEND_H     = 0.7
Z_TOL        = 0.03
V_MAX_H      = 0.4
V_MAX_Z      = 0.20
ORBIT_PERIOD = 6.0

CKPT_DIR = "trained_model/escort_follower_behavior"

# ─── 2.  FSM identical to drone firmware ────────────────────────
class FollowerFSM:
    def __init__(self):
        self.theta = 0.0
        self.prev_phase = -1
        self.landing_stage = 0
        self.target_apex = 0.0

    def _on_phase_change(self, ph, fp, lp):
        if ph == 0:  # hover → orbit initial angle
            rel = fp[:2] - lp[:2]
            if np.linalg.norm(rel) < 1e-3:
                rel = np.array([R_HOVER, 0.0])
            self.theta = np.arctan2(rel[1], rel[0])
        elif ph == 2:  # entering land
            self.landing_stage = 0
            self.target_apex = fp[2] + ASCEND_H

    def step_from_phase(self, ph, *, follower_pos, leader_pos,
                        leader_vel, dt):
        if ph != self.prev_phase:
            self._on_phase_change(ph, follower_pos, leader_pos)
            self.prev_phase = ph

        # --- desired target position ---------------------------
        if ph == 0:  # Hover → orbit
            self.theta = (self.theta + 2*np.pi*dt/ORBIT_PERIOD) % (2*np.pi)
            tgt_xy = leader_pos[:2] + R_HOVER * np.array(
                [np.cos(self.theta), np.sin(self.theta)])
            vz_corr = np.clip(0.5*(leader_pos[2]-follower_pos[2]),
                              -0.05, 0.05)
            target = np.array([*tgt_xy, follower_pos[2]+vz_corr])
            kp = KP_DEFAULT

        elif ph == 1:  # Move → trail
            offset = -leader_vel.copy(); offset[2] = 0
            if np.linalg.norm(offset) < 1e-3:
                offset = np.array([1, 0, 0])
            offset = FOLLOW_DIST * offset / np.linalg.norm(offset)
            target = leader_pos + offset
            kp = KP_DEFAULT

        else:  # ph == 2  Land
            kp = KP_LAND
            if self.landing_stage == 0:
                target = follower_pos.copy(); target[2] = self.target_apex
                if follower_pos[2] >= self.target_apex - Z_TOL:
                    self.landing_stage = 1
            else:
                target = follower_pos.copy(); target[2] = 0.0

        # PD controller (only P term here)
        cmd = kp * (target - follower_pos)
        cmd = np.clip(cmd, -MAX_ACT_VEL, MAX_ACT_VEL)

        # bumper to stay clear horizontally
        nxt = follower_pos + cmd*dt
        horiz = nxt - leader_pos; horiz[2] = 0
        if np.linalg.norm(horiz) < MIN_DIST and ph != 2:
            outward = horiz / (np.linalg.norm(horiz)+1e-6)
            cmd[:2] = outward[:2] * MAX_ACT_VEL

        # phase-dependent velocity caps
        if ph == 0:
            vx, vy, vz = cmd
        else:
            vx = cmd[0] * V_MAX_H
            vy = cmd[1] * V_MAX_H
            vz = cmd[2] * V_MAX_Z
        return np.array([vx, vy, vz])

# ─── 3. JAX environment (leader slowed) ─────────────────────────
@jdc.pytree_dataclass
class FBState(State):
    agents_locations:  jnp.ndarray
    agents_velocities: jnp.ndarray
    leader_location:   jnp.ndarray
    leader_velocity:   jnp.ndarray
    random_dir:        jnp.ndarray
    behavior_label:    int
    timestep:          int

class EscortFollowerBehavior(BaseParallelEnv):
    def __init__(self, init_flying_pos, init_leader_pos,
                 size=ENV_SIZE, steps_per_phase=STEPS_PER_PHASE):
        self.num_drones = init_flying_pos.shape[0]
        self.size = size
        self.steps_per_phase = steps_per_phase
        self.max_steps = 3*steps_per_phase + TAIL_STEPS
        self.possible_agents = ["agent_0"]
        self._init_agent_pos  = init_flying_pos
        self._init_leader_pos = init_leader_pos

    # ---- spaces ----
    def observation_space(self, _):
        low  = jnp.array([-self.size, -self.size, 0.0,
                          -MAX_ACT_VEL, -MAX_ACT_VEL, -MAX_ACT_VEL,
                          -self.size, -self.size, 0.0,
                          -MAX_ACT_VEL, -MAX_ACT_VEL, -0.2])
        return Box(low, -low, shape=(12,))

    def action_space(self, _):
        return Box(low=-MAX_ACT_VEL, high=MAX_ACT_VEL, shape=(3,))

    # ---- helpers ----
    @partial(jit, static_argnums=(0,))
    def _obs(self, st):
        return jnp.concatenate([st.agents_locations[0],
                                st.agents_velocities[0],
                                st.leader_location,
                                st.leader_velocity]).reshape(1,-1)

    def _sanitize(self, st, act):
        nxt = st.agents_locations + jnp.clip(act,
                                             -MAX_ACT_VEL,
                                             MAX_ACT_VEL)*SIM_DT
        lo = jnp.array([-self.size,-self.size,0.0])
        hi = jnp.array([ self.size, self.size,3.0])
        return jnp.clip(nxt, lo, hi)

    @partial(jit, static_argnums=(0,))
    def _transition(self, st, act, key):
        ph = jnp.clip((st.timestep-1)//self.steps_per_phase, 0, 2)

        vel_hov = jnp.zeros(3)
        vel_mov = jnp.concatenate([st.random_dir*LEADER_MOVE_SPEED,
                                   jnp.array([0.0])])
        vel_lnd = jnp.array([0.0, 0.0, -LEADER_DESCENT_RATE])
        l_vel   = jnp.select([ph==0, ph==1, ph==2],
                             [vel_hov, vel_mov, vel_lnd])

        l_loc = jnp.clip(st.leader_location + l_vel,
                         jnp.array([-self.size,-self.size,0.0]),
                         jnp.array([ self.size, self.size,3.0]))
        l_vel = jnp.where(l_loc[2]<=0.0, jnp.zeros_like(l_vel), l_vel)

        new_pos = self._sanitize(st, act)
        new_vel = new_pos - st.agents_locations

        return jdc.replace(st,
            agents_locations=new_pos,
            agents_velocities=new_vel,
            leader_location=l_loc,
            leader_velocity=l_vel,
            behavior_label=ph)

    # ---- API ----
    def reset(self, key):
        key, sub = jax.random.split(key)
        rnd = jax.random.uniform(sub,(2,), minval=-1.0, maxval=1.0)
        rnd = rnd/(jnp.linalg.norm(rnd)+1e-6)
        st = FBState(
            agents_locations=self._init_agent_pos,
            agents_velocities=jnp.zeros_like(self._init_agent_pos),
            leader_location=self._init_leader_pos,
            leader_velocity=jnp.zeros(3),
            random_dir=rnd,
            behavior_label=0,
            timestep=0)
        return self._obs(st), {}, st

    @partial(jit, static_argnums=(0,))
    def _step_core(self, st, act, key):
        st = jdc.replace(st, timestep=st.timestep+1)
        st = self._transition(st, act, key)
        obs   = self._obs(st)
        trunc = jnp.array([st.timestep >= self.max_steps])
        rew   = -jnp.linalg.norm(st.agents_locations -
                                 st.leader_location, axis=1)
        return obs, rew, jnp.zeros(1), trunc, st

    def step(self, st, act, key):
        return self._step_core(st, act, key)

# ─── 4. Actor – used only to classify phase ─────────────────────
class Actor(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(256)(x))
        mean = nn.Dense(self.act_dim,
                        kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        std = jnp.exp(log_std)
        logits = nn.Dense(3)(x)
        return mean, std, logits

# ─── 5.  Roll-out, metrics & plotting ───────────────────────────
def main():
    if not os.path.isdir(CKPT_DIR):
        raise FileNotFoundError(CKPT_DIR)

    # -------- PyBullet GUI --------------------------------------
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation(); p.setGravity(0,0,0); p.loadURDF("plane.urdf")
    f_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                rgbaColor=[1,0,0,1])
    l_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                rgbaColor=[1,1,0,1])
    f_id  = p.createMultiBody(baseMass=0, baseVisualShapeIndex=f_vis)
    l_id  = p.createMultiBody(baseMass=0, baseVisualShapeIndex=l_vis)

    # -------- Env & classifier ----------------------------------
    env = EscortFollowerBehavior(
        init_flying_pos=jnp.array([[0,0,1]], dtype=jnp.float32),
        init_leader_pos=jnp.array([1,1,1], dtype=jnp.float32))
    ACT_DIM = env.action_space(0).shape[0]
    actor   = Actor(ACT_DIM)
    params  = oc.PyTreeCheckpointer().restore(CKPT_DIR)["params"]

    @jax.jit
    def classify(p, o):
        _, _, logits = actor.apply({'params': p}, o)
        return jnp.argmax(logits, axis=-1)[0]

    fsm = FollowerFSM()
    rng = jax.random.PRNGKey(0)
    obs, _, st = env.reset(rng)

    # -------- METRIC BOOKKEEPING -------------------------------
    acc_correct = 0
    step_total  = 0
    act_mags    = []              # |v| each step

    follower_pos_hist, leader_pos_hist = [], []

    # -------- Roll-out loop ------------------------------------
    while st.timestep < env.max_steps:
        ph = int(classify(params, obs))

        vel = fsm.step_from_phase(
            ph,
            follower_pos=np.array(st.agents_locations[0]),
            leader_pos=np.array(st.leader_location),
            leader_vel=np.array(st.leader_velocity),
            dt=SIM_DT)

        # ---- accuracy / variance tracking
        true_ph = int(st.behavior_label)
        if ph == true_ph:
            acc_correct += 1
        step_total  += 1
        act_mags.append(np.linalg.norm(vel))

        act = jnp.asarray(vel, dtype=jnp.float32).reshape(1,3)
        rng, sub = jax.random.split(rng)
        obs, _, _, _, st = env.step(st, act, sub)

        follower_pos_hist.append(np.array(st.agents_locations[0]))
        leader_pos_hist.append(np.array(st.leader_location))

        p.resetBasePositionAndOrientation(
            f_id, st.agents_locations[0].tolist(), [0,0,0,1])
        p.resetBasePositionAndOrientation(
            l_id, st.leader_location.tolist(), [0,0,0,1])
        p.stepSimulation(); time.sleep(SIM_DT)

    p.disconnect()

    follower_pos_hist = np.array(follower_pos_hist)
    leader_pos_hist   = np.array(leader_pos_hist)

    # trajectories → velocities (finite difference)
    follower_vel = np.diff(follower_pos_hist, axis=0) / SIM_DT
    leader_vel   = np.diff(leader_pos_hist,   axis=0) / SIM_DT
    steps = np.arange(follower_vel.shape[0])

    # ── PLOT 1: 3-D trajectory --------------------------------
    fig = plt.figure(figsize=(7,6))
    ax  = fig.add_subplot(111, projection='3d')
    ax.plot(leader_pos_hist[:,0],  leader_pos_hist[:,1],
            leader_pos_hist[:,2], 'y-', label='Leader')
    ax.plot(follower_pos_hist[:,0], follower_pos_hist[:,1],
            follower_pos_hist[:,2], 'r-', label='Follower')
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title("3-D Trajectory"); ax.legend()
    ax.view_init(elev=30, azim=-60)

    # ── PLOT 2: velocity curves --------------------------------
    VEL_LIM = 1.0
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    ax1.plot(steps, follower_vel[:,0], label='Vx')
    ax1.plot(steps, follower_vel[:,1], label='Vy')
    ax1.plot(steps, follower_vel[:,2], label='Vz')
    ax1.set_title("Follower velocity"); ax1.set_ylabel("m/s")
    ax1.set_ylim(-VEL_LIM, VEL_LIM); ax1.legend()

    ax2.plot(steps, leader_vel[:,0], label='Vx')
    ax2.plot(steps, leader_vel[:,1], label='Vy')
    ax2.plot(steps, leader_vel[:,2], label='Vz')
    ax2.set_title("Leader velocity"); ax2.set_xlabel("Step")
    ax2.set_ylabel("m/s"); ax2.set_ylim(-VEL_LIM, VEL_LIM); ax2.legend()

    fig2.tight_layout()
    plt.show()

# ─── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
