# evaluate_follower_behavior.py  (velocity-logging version)
import time, csv, os
import matplotlib.pyplot as plt        # NEW
import jax, jax.numpy as jnp, numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from orbax import checkpoint as oc
import pybullet as p, pybullet_data
from datetime import datetime
from crazy_rl.multi_agent.jax.escort import EscortFollowerBehavior

# ------------------------------------------------------------------ #
# 1.  Environment & model as before
# ------------------------------------------------------------------ #
SIM_DT = 0.05                           # simulator timestep (s)  ← used for velocity
init_follower_pos = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
init_leader_pos   = jnp.array([1.0, 1.0, 1.0],  dtype=jnp.float32)
env = EscortFollowerBehavior(init_flying_pos=init_follower_pos,
                             init_leader_pos=init_leader_pos,
                             size=2.0,
                             steps_per_phase=50)
OBS_DIM = env.observation_space(0).shape[-1]
ACT_DIM = env.action_space(0).shape[0]

class Actor(nn.Module):
    act_dim: int
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(256)(x))
        mean   = nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
        log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
        logits = nn.Dense(3)(x)
        return mean, jnp.exp(log_std), logits

actor_model = Actor(ACT_DIM)
dummy_state = TrainState.create(
        apply_fn=actor_model.apply,
        params=actor_model.init(jax.random.PRNGKey(0), jnp.zeros((OBS_DIM,)))["params"],
        tx=optax.adam(1e-3))
ckpt_dir   = "trained_model/escort_follower_behavior"
actor_state = oc.PyTreeCheckpointer().restore(ckpt_dir, item=dummy_state)
params = actor_state.params

# ------------------------------------------------------------------ #
# 2.  CSV set-up (extra velocity columns)
# ------------------------------------------------------------------ #
log_filename = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file  = open(log_filename, "w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow([
    "Episode", "Step", "PredictedBehavior",
    "ActionX", "ActionY", "ActionZ",
    "FollowerX", "FollowerY", "FollowerZ",
    "LeaderX",   "LeaderY",   "LeaderZ",
    "FollowerVx","FollowerVy","FollowerVz",   # NEW
    "LeaderVx",  "LeaderVy",  "LeaderVz",     # NEW
    "TrueLabel"
])

# ------------------------------------------------------------------ #
# 3.  PyBullet visuals (unchanged)
# ------------------------------------------------------------------ #
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation(); p.setGravity(0,0,0); p.loadURDF("plane.urdf")
follower_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1,0,0,1])
leader_vis   = p.createVisualShape(p.GEOM_SPHERE, radius=0.15,rgbaColor=[1,1,0,1])
follower_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=follower_vis)
leader_body   = p.createMultiBody(baseMass=0, baseVisualShapeIndex=leader_vis)

behavior_names = {0:"Hover",1:"Move",2:"Land"}

# ------------------------------------------------------------------ #
# 4.  Evaluation loop (now keeps per-episode logs)
# ------------------------------------------------------------------ #
num_episodes = 3
for ep in range(1, num_episodes+1):
    print(f"\n=== Evaluation Episode {ep} ===")
    rng = jax.random.PRNGKey(ep)
    obs, info, state = env.reset(rng); obs = np.array(obs).reshape(-1)
    done, step_count = False, 0
    
    # --- per-episode buffers for plotting
    follower_pos_hist, leader_pos_hist = [], []
    
    while not done:
        mean, _, logits = actor_model.apply({'params': params}, jnp.array(obs))
        act        = np.array(mean)                  # deterministic mean action
        pred_class = int(np.argmax(np.array(logits)))
        
        # current positions
        follower_pos = np.array(state.agents_locations)[0]
        leader_pos   = np.array( state.leader_location)
        
        # store positions for velocity computation
        follower_pos_hist.append(follower_pos.copy())
        leader_pos_hist.append(  leader_pos.copy())
        
        # simple finite-difference to get velocity (zero for first step)
        if step_count==0:
            follower_vel = leader_vel = np.zeros(3)
        else:
            follower_vel = (follower_pos_hist[-1]-follower_pos_hist[-2])/SIM_DT
            leader_vel   = (leader_pos_hist[-1]  -leader_pos_hist[-2])  /SIM_DT
        
        # CSV log
        true_label = info.get("agent_0",{}).get("true_label",0)
        csv_writer.writerow([
            ep, step_count, behavior_names[pred_class],
            *act.tolist(),
            *follower_pos.tolist(), *leader_pos.tolist(),
            *follower_vel.tolist(),  *leader_vel.tolist(),
            true_label
        ])
        
        # step env
        act_jax = jnp.expand_dims(jnp.array(act,dtype=jnp.float32),0)
        rng, step_rng = jax.random.split(rng)
        obs_next, rew, term, trunc, info, state = env.step(state, act_jax, step_rng)
        obs      = np.array(obs_next).reshape(-1)
        done     = bool(np.array(term).squeeze() or np.array(trunc).squeeze())
        
        # update GUI
        p.resetBasePositionAndOrientation(follower_body, follower_pos, [0,0,0,1])
        p.resetBasePositionAndOrientation(leader_body,   leader_pos,   [0,0,0,1])
        p.stepSimulation(); time.sleep(0.1)
        step_count += 1
    
    print(f"Episode {ep} finished after {step_count} steps.")
    
    # -------------------------------------------------------------- #
    # 5.  Plot velocities for this episode
    # -------------------------------------------------------------- #
    follower_pos_hist = np.stack(follower_pos_hist)
    leader_pos_hist   = np.stack(leader_pos_hist)
    # finite-difference once more to have arrays of length N-1
    follower_vel_hist = np.diff(follower_pos_hist, axis=0)/SIM_DT
    leader_vel_hist   = np.diff(leader_pos_hist,   axis=0)/SIM_DT
    steps = np.arange(follower_vel_hist.shape[0])
    
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(steps, follower_vel_hist[:,0], label='Vx')
    plt.plot(steps, follower_vel_hist[:,1], label='Vy')
    plt.plot(steps, follower_vel_hist[:,2], label='Vz')
    plt.title(f"Follower velocity – episode {ep}")
    plt.ylabel("m/s"); plt.legend()
    
    plt.subplot(2,1,2)
    plt.plot(steps, leader_vel_hist[:,0], label='Vx')
    plt.plot(steps, leader_vel_hist[:,1], label='Vy')
    plt.plot(steps, leader_vel_hist[:,2], label='Vz')
    plt.title("Leader velocity")
    plt.xlabel("Step"); plt.ylabel("m/s"); plt
