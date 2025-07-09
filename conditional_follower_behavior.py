# conditional_follower_behavior.py
import os
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import jit
from flax import linen as nn
from orbax import checkpoint as oc

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State
from crazy_rl.utils.jax_spaces import Box, Space

import pybullet as p
import pybullet_data

# ────────────────────────────────────────────────────────────────
# 1.  Constants (must match training)
# ────────────────────────────────────────────────────────────────
SIM_DT            = 0.05   # s
STEPS_PER_PHASE   = 80     # 3 phases * 80 ticks ≈ 12 s
MAX_ACT_VEL       = 1.0    # m/s
ENV_SIZE          = 2.0    # m (workspace half-width)
CKPT_DIR          = "trained_model/escort_follower_behavior"

# ----------------------------------------------------------------
# 2.  Finite-state follower controller (geometry only)
# ----------------------------------------------------------------
class FollowerFSM:
    R_HOVER     = 1.0
    FOLLOW_DIST = 1.0
    MIN_DIST    = 0.5

    KP_DEFAULT = 2.0
    KP_LAND    = 1.2
    ASCEND_H   = 0.7
    Z_TOL      = 0.03

    def __init__(self):
        self.theta = 0.0
        self.prev_phase = -1
        self.landing_stage = 0        # 0 = climb, 1 = descend
        self.target_apex = 0.0

    # ---- helper called on phase switch -------------------------
    def _on_phase_change(self, phase, follower_pos, leader_pos):
        if phase == 0:  # HOVER
            rel = follower_pos[:2] - leader_pos[:2]
            if np.linalg.norm(rel) < 1e-3:
                rel = np.array([self.R_HOVER, 0.0])
            self.theta = np.arctan2(rel[1], rel[0])
        elif phase == 2:  # LAND
            self.landing_stage = 0
            self.target_apex = follower_pos[2] + self.ASCEND_H

    # ---- main step interface ------------------------------------
    def step_from_phase(self, phase, *, follower_pos, leader_pos,
                        leader_vel, dt):

        # Detect phase switch
        if phase != self.prev_phase:
            self._on_phase_change(phase, follower_pos, leader_pos)
            self.prev_phase = phase

        # ─── phase-specific target computation ───────────────────
        if phase == 0:  # HOVER
            self.theta = (self.theta + 2*np.pi*dt/4) % (2*np.pi)
            target = leader_pos + np.array(
                [self.R_HOVER*np.cos(self.theta),
                 self.R_HOVER*np.sin(self.theta), 0.0])
            kp = self.KP_DEFAULT

        elif phase == 1:  # MOVE
            offset = -leader_vel.copy(); offset[2] = 0.0
            if np.linalg.norm(offset) < 1e-3:
                offset = np.array([1.0, 0.0, 0.0])
            offset = self.FOLLOW_DIST*offset/np.linalg.norm(offset)
            target = leader_pos + offset
            kp = self.KP_DEFAULT

        else:            # LAND (ascend → descend)
            kp = self.KP_LAND
            if self.landing_stage == 0:
                target = follower_pos.copy()
                target[2] = self.target_apex
                if follower_pos[2] >= self.target_apex - self.Z_TOL:
                    self.landing_stage = 1
            else:
                target = follower_pos.copy()
                target[2] = 0.0

        # ─── proportional control + safety bumper ────────────────
        cmd_vel = kp * (target - follower_pos)
        cmd_vel = np.clip(cmd_vel, -MAX_ACT_VEL, MAX_ACT_VEL)

        next_pos = follower_pos + cmd_vel*dt
        horiz = next_pos - leader_pos; horiz[2] = 0.0
        dist = np.linalg.norm(horiz)
        if dist < self.MIN_DIST and phase != 2:
            outward = horiz / (dist + 1e-6)
            cmd_vel[:2] = outward[:2] * MAX_ACT_VEL

        return cmd_vel

# ----------------------------------------------------------------
# 3.  Environment (identical to training)
# ----------------------------------------------------------------
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
    """Leader script: hover → move → land."""

    # ---- init ---------------------------------------------------
    def __init__(self, init_flying_pos, init_leader_pos,
                 size=ENV_SIZE, steps_per_phase=STEPS_PER_PHASE):
        self.num_drones      = init_flying_pos.shape[0]
        self.size            = size
        self.steps_per_phase = steps_per_phase
        self.max_steps       = 3 * steps_per_phase + 80
        self.possible_agents = ["agent_0"]
        self._init_agent_pos  = init_flying_pos
        self._init_leader_pos = init_leader_pos

    # ---- spaces -------------------------------------------------
    def observation_space(self, agent):
        low  = jnp.array([-self.size, -self.size, 0.0,
                          -MAX_ACT_VEL, -MAX_ACT_VEL, -MAX_ACT_VEL,
                          -self.size, -self.size, 0.0,
                          -MAX_ACT_VEL, -MAX_ACT_VEL, -0.2])
        high = -low
        return Box(low, high, shape=(12,))

    def action_space(self, agent):
        return Box(low=-MAX_ACT_VEL, high=MAX_ACT_VEL, shape=(3,))

    # ---- obs helper --------------------------------------------
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state):
        return jnp.concatenate(
            [state.agents_locations[0],
             state.agents_velocities[0],
             state.leader_location,
             state.leader_velocity]).reshape(1, -1)

    # ---- action sanitiser --------------------------------------
    def _sanitize_action(self, state, actions):
        new_pos = state.agents_locations + \
                  jnp.clip(actions, -MAX_ACT_VEL, MAX_ACT_VEL) * SIM_DT
        low  = jnp.array([-self.size, -self.size, 0.0])
        high = jnp.array([ self.size,  self.size, 3.0])
        return jnp.clip(new_pos, low, high)

    # ---- leader motion + state transition ----------------------
    @partial(jit, static_argnums=(0,))
    def _transition_state(self, state, actions, key):
        phase = jnp.clip((state.timestep - 1)//self.steps_per_phase, 0, 2)

        vel_hover = jnp.zeros(3)
        vel_move  = jnp.concatenate([state.random_dir*0.1, jnp.array([0.0])])
        vel_land  = jnp.array([0.0, 0.0, -0.1])
        leader_vel = jnp.select([phase==0, phase==1, phase==2],
                                [vel_hover, vel_move, vel_land])

        leader_loc = jnp.clip(state.leader_location + leader_vel,
                              jnp.array([-self.size, -self.size, 0.0]),
                              jnp.array([ self.size,  self.size, 3.0]))

        leader_vel = jnp.where(leader_loc[2] <= 0.0,
                               jnp.zeros_like(leader_vel),
                               leader_vel)

        new_pos = self._sanitize_action(state, actions)
        new_vel = new_pos - state.agents_locations

        return jdc.replace(
            state,
            agents_locations=new_pos,
            agents_velocities=new_vel,
            leader_location=leader_loc,
            leader_velocity=leader_vel,
            behavior_label=phase)

    # ---- reset --------------------------------------------------
    def reset(self, key):
        key, sub = jax.random.split(key)
        rnd_dir = jax.random.uniform(sub, (2,), minval=-1., maxval=1.)
        rnd_dir = rnd_dir / (jnp.linalg.norm(rnd_dir)+1e-6)

        state = FBState(
            agents_locations=self._init_agent_pos,
            agents_velocities=jnp.zeros_like(self._init_agent_pos),
            leader_location=self._init_leader_pos,
            leader_velocity=jnp.zeros(3),
            random_dir=rnd_dir,
            behavior_label=0,
            timestep=0
        )
        obs = self._compute_obs(state)
        return obs, {}, state

    # ---- step ---------------------------------------------------
    @partial(jit, static_argnums=(0,))
    def _step_core(self, state, actions, key):
        state = jdc.replace(state, timestep=state.timestep+1)
        state = self._transition_state(state, actions, key)
        obs  = self._compute_obs(state)
        trnc = (state.timestep >= self.max_steps)
        rew  = -jnp.linalg.norm(state.agents_locations -
                                state.leader_location, axis=1)
        return obs, rew, jnp.zeros(1), jnp.array([trnc]), state

    def step(self, state, actions, key):
        return self._step_core(state, actions, key)

# ----------------------------------------------------------------
# 4.  Neural phase-classifier (same as in training)
# ----------------------------------------------------------------
class Actor(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(256)(x))
        mean = nn.Dense(self.act_dim,
                        kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = self.param("log_std", nn.initializers.zeros,
                             (self.act_dim,))
        std = jnp.exp(log_std)
        logits = nn.Dense(3)(x)     # phase classifier
        return mean, std, logits

# ----------------------------------------------------------------
# 5.  Main demo with trajectory logging
# ----------------------------------------------------------------
def main():
    if not os.path.isdir(CKPT_DIR):
        raise FileNotFoundError(f"Checkpoint folder '{CKPT_DIR}' not found.")

    # ---------- PyBullet GUI set-up ------------------------------
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, 0)
    p.loadURDF("plane.urdf")

    f_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1,
                                rgbaColor=[1, 0, 0, 1])
    l_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,
                                rgbaColor=[1, 1, 0, 1])
    f_id = p.createMultiBody(baseVisualShapeIndex=f_vis, baseMass=0)
    l_id = p.createMultiBody(baseVisualShapeIndex=l_vis, baseMass=0)

    # ---------- Env & policy -------------------------------------
    env = EscortFollowerBehavior(
        init_flying_pos=jnp.array([[0., 0., 1.]], dtype=jnp.float32),
        init_leader_pos=jnp.array([1., 1., 1.],  dtype=jnp.float32))

    ACT_DIM = env.action_space(0).shape[0]
    actor = Actor(ACT_DIM)

    ckpt = oc.PyTreeCheckpointer()
    actor_params = ckpt.restore(CKPT_DIR)["params"]

    @jax.jit
    def classify(params, obs):
        _, _, logits = actor.apply({'params': params}, obs)
        return jnp.argmax(logits, axis=-1)[0]  # 0 / 1 / 2

    fsm = FollowerFSM()

    # ---------- Rollout ------------------------------------------
    key = jax.random.PRNGKey(0)
    obs, _, state = env.reset(key)

    follower_traj, leader_traj = [], []

    while state.timestep < env.max_steps:
        phase = int(classify(actor_params, obs))

        cmd_vel = fsm.step_from_phase(
            phase,
            follower_pos=np.array(state.agents_locations[0]),
            leader_pos=np.array(state.leader_location),
            leader_vel=np.array(state.leader_velocity),
            dt=SIM_DT)

        act = jnp.asarray(cmd_vel, dtype=jnp.float32).reshape(1, 3)

        key, sub = jax.random.split(key)
        obs, _, _, _, state = env.step(state, act, sub)

        # --- log positions
        follower_traj.append(np.array(state.agents_locations[0]))
        leader_traj.append(np.array(state.leader_location))

        # --- update GUI
        p.resetBasePositionAndOrientation(
            f_id, follower_traj[-1].tolist(), [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(
            l_id, leader_traj[-1].tolist(),   [0, 0, 0, 1])
        p.stepSimulation()
        time.sleep(SIM_DT)

    # ---------- save trajectories --------------------------------
    np.save("follower_traj.npy", np.asarray(follower_traj))
    np.save("leader_traj.npy",   np.asarray(leader_traj))
    print("Trajectories saved as follower_traj.npy & leader_traj.npy")

    time.sleep(2)
    p.disconnect()

# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
