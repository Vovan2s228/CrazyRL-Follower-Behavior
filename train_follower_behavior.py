# train_follower_behavior.py
"""Train a follower behavior policy in a PyBullet simulation (non-vectorized PPO)."""
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
from orbax import checkpoint as oc
import pybullet as p
import pybullet_data

from crazy_rl.multi_agent.jax.escort import EscortFollowerBehavior

# ---------------- hyper-parameters ---------------- #
N_ENVS        = 1
ROLLOUT_LEN   = 256
TOTAL_STEPS   = 200_000
GAMMA         = 0.99
LAM           = 0.95
CLIP          = 0.2
VF_COEF       = 0.5
ENT_COEF      = 0.01
CLS_COEF      = 1.0
BONUS         = 0.2
PPO_EPOCHS    = 4
MINIBATCH_SIZE= 256

# ---------------- environment setup ---------------- #
init_follower_pos = jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32)
init_leader_pos   = jnp.array([1.0, 1.0, 1.0],  dtype=jnp.float32)
env = EscortFollowerBehavior(init_flying_pos=init_follower_pos,
                             init_leader_pos=init_leader_pos,
                             size=2.0,
                             steps_per_phase=80)  # Increased steps_per_phase to extend hover phase
OBS_DIM = env.observation_space(0).shape[-1]
ACT_DIM = env.action_space(0).shape[0]

# Initialize PyBullet simulation (visualization)
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, 0)
p.loadURDF("plane.urdf")
follower_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])   # red sphere (follower)
leader_visual   = p.createVisualShape(p.GEOM_SPHERE, radius=0.15, rgbaColor=[1, 1, 0, 1])  # yellow sphere (leader)
follower_body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=follower_visual, basePosition=[0, 0, 0])
leader_body   = p.createMultiBody(baseMass=0, baseVisualShapeIndex=leader_visual,   basePosition=[0, 0, 0])

# ---------------- network definitions ---------------- #
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

class Critic(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.tanh(nn.Dense(256)(x))
        x = nn.tanh(nn.Dense(256)(x))
        return nn.Dense(1)(x).squeeze(-1)

# ---- initialization ----
key = jax.random.PRNGKey(0)
actor_model = Actor(ACT_DIM)
critic_model = Critic()
key, actor_key, critic_key = jax.random.split(key, 3)
actor_state = TrainState.create(
    apply_fn = actor_model.apply,
    params   = actor_model.init(actor_key, jnp.zeros((OBS_DIM,)))['params'],
    tx       = optax.adam(3e-4),
)
critic_state = TrainState.create(
    apply_fn = critic_model.apply,
    params   = critic_model.init(critic_key, jnp.zeros((OBS_DIM,)))['params'],
    tx       = optax.adam(3e-4),
)

@jax.jit
def sample_action(params, obs, rng):
    mean, std, logits = actor_model.apply({'params': params}, obs)
    noise = jax.random.normal(rng, shape=mean.shape)
    act = mean + noise * std
    logp = -0.5 * jnp.sum(noise**2 + jnp.log(2 * jnp.pi * std**2), axis=-1)
    pred = jnp.argmax(logits, axis=-1)
    return act, logp, pred

# ---- buffers ----
buf_obs    = np.zeros((ROLLOUT_LEN, N_ENVS, OBS_DIM), np.float32)
buf_acts   = np.zeros((ROLLOUT_LEN, N_ENVS, ACT_DIM), np.float32)
buf_logp   = np.zeros((ROLLOUT_LEN, N_ENVS), np.float32)
buf_vals   = np.zeros((ROLLOUT_LEN, N_ENVS), np.float32)
buf_rews   = np.zeros((ROLLOUT_LEN, N_ENVS), np.float32)
buf_dones  = np.zeros((ROLLOUT_LEN, N_ENVS), np.float32)
buf_labels = np.zeros((ROLLOUT_LEN, N_ENVS), np.int32)

# Reset environment to start
master_key = key
reset_key, master_key = jax.random.split(master_key)
obs, info, state = env.reset(reset_key)
obs = np.array(obs).reshape(-1)  # flatten observation to shape (OBS_DIM,)
step_idx = 0
acc_count = 0
prev_done = False

# Training loop
for global_step in range(TOTAL_STEPS):
    # Sample action from the policy
    master_key, rng = jax.random.split(master_key)
    act, logp, pred = sample_action(actor_state.params, obs, rng)
    act = np.array(act)                      # action as numpy array
    logp = float(np.array(logp))             # log-probability (scalar)
    pred_class = int(np.array(pred))         # predicted behavior class
    true_label = info.get("agent_0", {}).get("true_label", 0)  # actual behavior label from env
    if prev_done:
        # Override label after reset (new episode start) to hover (0)
        true_label = 0
    # Update accuracy count
    if pred_class == true_label:
        acc_count += 1
        correct = True
    else:
        correct = False

    # Take environment step with the selected action
    act_jax = jnp.expand_dims(jnp.array(act, dtype=jnp.float32), axis=0)  # shape (1, ACT_DIM)
    step_key, master_key = jax.random.split(master_key)
    obs_next, rew, terminated, truncated, info, state = env.step(state, act_jax, step_key)
    obs_next = np.array(obs_next).reshape(-1)
    done_flag = bool(np.array(terminated).squeeze() or np.array(truncated).squeeze())
    reward = float(np.array(rew).squeeze())
    if correct:
        # Add bonus reward for correct classification
        reward += BONUS

    # Store transition in buffers
    buf_obs[step_idx, 0]   = obs
    buf_acts[step_idx, 0]  = act
    buf_logp[step_idx, 0]  = logp
    buf_vals[step_idx, 0]  = float(np.array(critic_model.apply({'params': critic_state.params}, jnp.array(obs))))
    buf_rews[step_idx, 0]  = reward
    buf_dones[step_idx, 0] = float(done_flag)
    buf_labels[step_idx, 0] = true_label

    # Update visualization (move follower and leader in PyBullet)
    follower_pos = np.array(state.agents_locations)[0]
    leader_pos   = np.array(state.leader_location)
    p.resetBasePositionAndOrientation(follower_body, follower_pos.tolist(), [0,0,0,1])
    p.resetBasePositionAndOrientation(leader_body, leader_pos.tolist(), [0,0,0,1])
    p.stepSimulation()

    # Prepare next step
    obs = obs_next
    prev_done = done_flag
    step_idx += 1

    # If episode ended, reset environment and continue (auto-reset logic)
    if done_flag:
        reset_key, master_key = jax.random.split(master_key)
        obs, info, state = env.reset(reset_key)
        obs = np.array(obs).reshape(-1)
        # (Keep prev_done=True so that next step's label is overridden to 0)

    # If rollout buffer is full, perform PPO update
    if step_idx == ROLLOUT_LEN:
        # Compute GAE (generalized advantage estimation) for advantage buffer
        next_val = float(np.array(critic_model.apply({'params': critic_state.params}, jnp.array(obs))))
        adv_buf = np.zeros(ROLLOUT_LEN, np.float32)
        gae = 0.0
        for t in reversed(range(ROLLOUT_LEN)):
            mask = 1.0 - buf_dones[t, 0]  # 0 if episode ended at t, else 1
            next_value = next_val if t == ROLLOUT_LEN - 1 else buf_vals[t+1, 0]
            delta = buf_rews[t, 0] + GAMMA * next_value * mask - buf_vals[t, 0]
            gae = delta + GAMMA * LAM * mask * gae
            adv_buf[t] = gae
            next_val = buf_vals[t, 0]
        # Advantage normalization
        adv_buf = (adv_buf - adv_buf.mean()) / (adv_buf.std() + 1e-8)
        ret_buf = adv_buf + buf_vals[:, 0]  # target returns

        # Convert buffers to JAX arrays for vectorized computation
        mb_obs   = jnp.array(buf_obs)
        mb_act   = jnp.array(buf_acts)
        mb_oldlp = jnp.array(buf_logp)
        mb_adv   = jnp.array(adv_buf)
        mb_ret   = jnp.array(ret_buf)
        mb_lbl   = jnp.array(buf_labels)

        total_pg = total_vf = total_ent = total_cls = total_loss = 0.0
        count = 0

        # Define combined PPO + classification loss
        def loss_fn(actor_params, critic_params, obs_b, act_b, oldlp_b, adv_b, ret_b, lbl_b):
            mean, std, logits = actor_model.apply({'params': actor_params}, obs_b)
            new_logp = -0.5 * jnp.sum(((act_b - mean) / std)**2 + 2 * jnp.log(std) + jnp.log(2 * jnp.pi), axis=-1)
            ratio = jnp.exp(new_logp - oldlp_b)
            pg_loss1 = adv_b * ratio
            pg_loss2 = adv_b * jnp.clip(ratio, 1.0 - CLIP, 1.0 + CLIP)
            pg_loss = -jnp.mean(jnp.minimum(pg_loss1, pg_loss2))
            # Value (critic) loss (Huber)
            value_pred = critic_model.apply({'params': critic_params}, obs_b)
            vf_loss = jnp.mean((ret_b - value_pred)**2)
            # Entropy bonus
            ent_loss = -jnp.mean(0.5 * (1 + jnp.log(2 * jnp.pi * (std**2))) + jnp.log(std))
            # Classification loss (cross-entropy with true labels)
            labels_onehot = jax.nn.one_hot(lbl_b, num_classes=3)
            log_probs = jax.nn.log_softmax(logits)
            cls_loss = -jnp.mean(jnp.sum(labels_onehot * log_probs, axis=1))
            # Total loss (weighted sum)
            total_loss = pg_loss + VF_COEF * vf_loss - ENT_COEF * ent_loss + CLS_COEF * cls_loss
            return total_loss, (pg_loss, vf_loss, ent_loss, cls_loss)

        loss_grad_fn = jax.jit(jax.value_and_grad(loss_fn, argnums=(0,1), has_aux=True))

        # Shuffle indices for minibatch sampling
        total_samples = ROLLOUT_LEN * N_ENVS  # here N_ENVS=1, so total_samples=ROLLOUT_LEN
        indices = np.arange(total_samples)
        np.random.shuffle(indices)
        # Optimize policy for multiple epochs
        for _ in range(PPO_EPOCHS):
            for start in range(0, total_samples, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                batch_idx = indices[start:end]
                # Flatten batch observations/actions
                b_obs   = mb_obs.reshape(-1, OBS_DIM)[batch_idx]
                b_act   = mb_act.reshape(-1, ACT_DIM)[batch_idx]
                b_oldlp = mb_oldlp.reshape(-1)[batch_idx]
                b_adv   = mb_adv.reshape(-1)[batch_idx]
                b_ret   = mb_ret.reshape(-1)[batch_idx]
                b_lbl   = mb_lbl.reshape(-1)[batch_idx]
                # Compute loss and gradients
                (loss_val, (pg, vf, ent, cls)), (grad_a, grad_c) = loss_grad_fn(
                    actor_state.params, critic_state.params,
                    b_obs, b_act, b_oldlp, b_adv, b_ret, b_lbl
                )
                # Apply gradient updates
                actor_state  = actor_state.apply_gradients(grads=grad_a)
                critic_state = critic_state.apply_gradients(grads=grad_c)
                total_pg += float(pg);  total_vf += float(vf)
                total_ent += float(ent); total_cls += float(cls)
                total_loss += float(loss_val); count += 1

        # Log training metrics
        acc = acc_count / (ROLLOUT_LEN * N_ENVS)
        mean_rew = buf_rews.sum() / (ROLLOUT_LEN * N_ENVS)
        print(f"[Step {global_step+1:6d}] acc={acc:.2%}  rew={mean_rew:.3f}  "
              f"loss: total={total_loss/count:.4f} pg={total_pg/count:.4f} "
              f"vf={total_vf/count:.4f} ent={total_ent/count:.4f} cls={total_cls/count:.4f}")

        # Reset rollout buffer and counters
        step_idx = 0
        acc_count = 0

# Save trained actor model for evaluation
import shutil
ckpt_dir = "trained_model/escort_follower_behavior"
if os.path.exists(ckpt_dir):
    shutil.rmtree(ckpt_dir)
os.makedirs(os.path.dirname(ckpt_dir), exist_ok=True)
oc.PyTreeCheckpointer().save(ckpt_dir, actor_state)
