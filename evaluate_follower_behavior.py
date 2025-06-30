# evaluate_follower_behavior.py
import time, csv
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from orbax import checkpoint as oc
import pybullet as p
import pybullet_data
from datetime import datetime
from crazy_rl.multi_agent.jax.escort import EscortFollowerBehavior

# --- Setup ---
init_follower_pos = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
init_leader_pos   = jnp.array([1.0, 1.0, 1.0],  dtype=jnp.float32)
env = EscortFollowerBehavior(init_flying_pos=init_follower_pos,
                             init_leader_pos=init_leader_pos,
                             size=2.0,
                             steps_per_phase=50)
OBS_DIM = env.observation_space(0).shape[-1]
ACT_DIM = env.action_space(0).shape[0]

# --- Actor Definition ---
class Actor(nn.Module):
	act_dim: int
	@nn.compact
	def __call__(self, x):
		x = nn.tanh(nn.Dense(256)(x))
		x = nn.tanh(nn.Dense(256)(x))
		mean = nn.Dense(self.act_dim, kernel_init=nn.initializers.orthogonal(0.01))(x)
		log_std = self.param("log_std", nn.initializers.zeros, (self.act_dim,))
		std = jnp.exp(log_std)
		logits = nn.Dense(3)(x)
		return mean, std, logits

# --- Load Trained Parameters ---
actor_model = Actor(ACT_DIM)
dummy_state = TrainState.create(
	apply_fn=actor_model.apply,
	params=actor_model.init(jax.random.PRNGKey(0), jnp.zeros((OBS_DIM,)))["params"],
	tx=optax.adam(1e-3)
)
ckpt_dir = "trained_model/escort_follower_behavior"
actor_state = oc.PyTreeCheckpointer().restore(ckpt_dir, item=dummy_state)
params = actor_state.params

# --- CSV Setup ---
log_filename = f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, mode="w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow([
	"Episode", "Step", "PredictedBehavior",
	"ActionX", "ActionY", "ActionZ",
	"FollowerX", "FollowerY", "FollowerZ",
	"LeaderX", "LeaderY", "LeaderZ",
	"TrueLabel"
])

# --- PyBullet Setup ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, 0)
p.loadURDF("plane.urdf")
follower_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
leader_visual   = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[1, 1, 0, 1])
follower_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=follower_visual, basePosition=[0,0,0])
leader_body   = p.createMultiBody(baseMass=0, baseVisualShapeIndex=leader_visual,   basePosition=[0,0,0])

behavior_names = {0: "Hover", 1: "Move", 2: "Land"}

# --- Evaluation Loop ---
num_episodes = 3
for ep in range(1, num_episodes + 1):
	print(f"=== Evaluation Episode {ep} ===")
	rng = jax.random.PRNGKey(ep)
	obs, info, state = env.reset(rng)
	obs = np.array(obs).reshape(-1)
	done = False
	step_count = 0

	while not done:
		mean, std, logits = actor_model.apply({'params': params}, jnp.array(obs))
		mean = np.array(mean)
		logits = np.array(logits)
		pred_class = int(np.argmax(logits))
		act = mean

		# Get positions
		follower_pos = np.array(state.agents_locations)[0].tolist()
		leader_pos   = np.array(state.leader_location).tolist()

		# Logging
		true_label = info.get("agent_0", {}).get("true_label", 0)
		csv_writer.writerow([
			ep, step_count, behavior_names[pred_class],
			*act.tolist(), *follower_pos, *leader_pos, true_label
		])
		print(f"Step {step_count:3d}: Predicted = {behavior_names[pred_class]}  Action = {act}  Follower = {follower_pos}")

		# Environment Step
		act_jax = jnp.expand_dims(jnp.array(act, dtype=jnp.float32), axis=0)
		rng, step_rng = jax.random.split(rng)
		obs_next, rew, terminated, truncated, info, state = env.step(state, act_jax, step_rng)
		obs = np.array(obs_next).reshape(-1)
		done = bool(np.array(terminated).squeeze() or np.array(truncated).squeeze())

		# Update visuals
		p.resetBasePositionAndOrientation(follower_body, follower_pos, [0,0,0,1])
		p.resetBasePositionAndOrientation(leader_body, leader_pos, [0,0,0,1])
		p.stepSimulation()
		time.sleep(0.1)
		step_count += 1

	print(f"Episode {ep} finished after {step_count} steps.\n")

p.disconnect()
log_file.close()
print(f"Log saved to: {log_filename}")
