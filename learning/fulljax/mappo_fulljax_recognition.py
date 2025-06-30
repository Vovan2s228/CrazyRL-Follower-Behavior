import argparse
import os
import time
from distutils.util import strtobool
from typing import NamedTuple, Sequence, Tuple, Optional, List

import chex
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import vmap
from tqdm import tqdm

# Import CrazyRL environment and wrappers
from crazy_rl.multi_agent.jax.behavior_recognition import BehaviorRecognitionJax
from crazy_rl.utils.jax_wrappers import (
    AddIDToObs,
    AutoReset,
    ClipActions,
    LogWrapper,
    NormalizeObservation,
    NormalizeVecReward,
    VecEnv,
)

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--debug", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
                        help="run in debug mode")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=128, help="the number of parallel environments")
    parser.add_argument("--num-steps", type=int, default=10, help="the number of steps per rollout (trajectory length)")
    parser.add_argument("--total-timesteps", type=int, default=int(1e6),
                        help="total timesteps of the experiment")
    parser.add_argument("--update-epochs", type=int, default=2, help="the number of epochs to update the policy")
    parser.add_argument("--num-minibatches", type=int, default=2, help="the number of minibatches for PPO update")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor gamma")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--gae-lambda", type=float, default=0.99, help="lambda for GAE")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="ppo clip epsilon")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="entropy bonus coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.8, help="value function loss coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max norm for gradient clipping")
    parser.add_argument("--activation", type=str, default="tanh", help="activation function (tanh or relu)")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="whether to anneal the learning rate linearly")
    args = parser.parse_args()
    # fmt: on
    return args

class Actor(nn.Module):
    action_dim: int  # For discrete actions, number of actions; for continuous, dimension of action vector
    activation: str = "tanh"
    @nn.compact
    def __call__(self, obs_input: jnp.ndarray):
        # Two hidden layers
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs_input)
        x = act_fn(x)
        x = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act_fn(x)
        # Output layer
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        # If action_dim == 1 (e.g., continuous scalar) or >1 (discrete or multi-continuous)
        # Determine distribution: if discrete action space (BehaviorRecognition uses Discrete(n))
        if isinstance(self.action_dim, int) and self.action_dim > 1:
            # Use categorical distribution for discrete actions
            return distrax.Categorical(logits=logits)
        else:
            # Continuous action(s): use Gaussian with state-independent log-std
            # Ensure action_dim is a sequence length for continuous scenario
            action_dim = self.action_dim if isinstance(self.action_dim, int) else self.action_dim[0]
            log_std = self.param("log_std", nn.initializers.zeros, (action_dim,))
            return distrax.MultivariateNormalDiag(loc=logits, scale_diag=jnp.exp(log_std))

class Critic(nn.Module):
    activation: str = "tanh"
    @nn.compact
    def __call__(self, global_obs: jnp.ndarray):
        act_fn = nn.relu if self.activation == "relu" else nn.tanh
        v = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(global_obs)
        v = act_fn(v)
        v = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(v)
        v = act_fn(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)
        return jnp.squeeze(v, axis=-1)  # output value as scalar

# Define Transition tuple for trajectory storage
class Transition(NamedTuple):
    terminated: jnp.ndarray            # shape (num_envs,)
    joint_actions: jnp.ndarray         # shape (num_envs, num_agents, action_dim)
    value: jnp.ndarray                # shape (num_envs,)
    reward: jnp.ndarray               # shape (num_envs,)
    log_prob: jnp.ndarray             # shape (num_envs, num_agents)
    obs: jnp.ndarray                  # shape (num_envs, num_agents, obs_dim)
    global_obs: jnp.ndarray           # shape (num_envs, global_obs_dim)
    info: Optional[dict] = None                      # info dict with possible episode metrics

def make_train(args):
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)
    minibatch_size = args.num_envs * args.num_steps // args.num_minibatches

    def train(key: chex.PRNGKey, lr: Optional[float] = None):
        # Initialize environment (BehaviorRecognition with CrazyRL JAX)
        num_drones = 1  # one learning agent (the observer)
        env = BehaviorRecognitionJax(
            init_agent_pos=jnp.array([[0.0, 0.0, 1.0]]),   # observer initial position
            init_peer_pos=jnp.array([1.0, 1.0, 1.5]),      # peer drone initial position
            size=2.0,
            max_steps=50
        )
        # Wrap environment with required wrappers
        #env = ClipActions(env)
        env = NormalizeObservation(env)
        env = AddIDToObs(env, num_drones)        # add one-hot agent ID to observation
        env = LogWrapper(env)
        env = NormalizeVecReward(env, args.gamma)
        env = AutoReset(env)
        env = VecEnv(env)

        # Initial reset of vectorized envs
        reset_keys = jax.random.split(jax.random.PRNGKey(args.seed), args.num_envs)
        obs, info, env_states = env.reset(reset_keys)
        # obs shape: (num_envs, num_drones, obs_dim(+id))
        # env_states is a NormalizeVecRewEnvState (with env_state inside)

        if args.debug:
            # Print initial true behavior labels of peer for each environment
            try:
                true_labels = env_states.env_state.behavior_label  # shape (num_envs,)
            except AttributeError:
                # If env_states is not wrapped by NormalizeVecReward (no behavior_label), handle differently
                true_labels = getattr(env_states, "behavior_label", None)
            print("Initial true behavior class labels:", np.array(true_labels))

        # Initialize policy and value networks
        actor = Actor(env.action_space(0).n, activation=args.activation)   # number of discrete actions
        critic = Critic(activation=args.activation)
        key, actor_key, critic_key = jax.random.split(key, 3)
        # Determine observation and state dimensions for network initialization
        obs_dim = obs.shape[-1]
        global_state_dim = env.state(env_states).shape[1]
        print("Global observation shape:", env.state(env_states).shape)
        print("obs shape passed to actor.apply:", obs.shape)  # Should be (num_envs, obs_dim)
        print("obs[0] shape:", obs[0].shape)  # Should be (num_envs, obs_dim)print("obs[0] shape:", obs[0].shape)
        dummy_local_obs = jnp.zeros((obs.shape[-1],))
        
        dummy_global_obs = jnp.zeros((global_state_dim,))
        actor_params = actor.init(actor_key, dummy_local_obs)
        critic_params = critic.init(critic_key, dummy_global_obs)
        # Set up optimizer
        if args.anneal_lr:
            # linear learning rate schedule
            def linear_schedule(count):
                frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / num_updates
                return args.lr * frac
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(args.lr, eps=1e-5),
            )
        # Initialize training state (parameters + optimizer state)
        actor_state = TrainState.create(apply_fn=actor.apply, params=actor_params, tx=tx)
        critic_state = TrainState.create(apply_fn=critic.apply, params=critic_params, tx=tx)

        # Helper to get policy distributions for all agents
        def _ma_get_pi(params, obs: jnp.ndarray) -> List[distrax.Distribution]:
            # obs shape: (num_envs, num_drones, obs_dim_with_id)
            # Return list of length num_drones, each a distribution over actions for all envs
            return [actor.apply(params, obs[:, i, :]) for i in range(num_drones)]

        # Helper to sample actions and log probs from distributions for all agents
        def _ma_sample_and_log_prob_from_pi(pi_list: List[distrax.Distribution], rng_key: chex.PRNGKey):
            # rng_key will be split for each agent
            subkeys = jax.random.split(rng_key, num_drones)
            actions_list = []
            log_probs_list = []
            for i in range(num_drones):
                dist = pi_list[i]  # distribution for agent i across all envs
                sample, logprob = dist.sample_and_log_prob(seed=subkeys[i])
                # Ensure sample has shape (num_envs, action_dim_per_agent)
                if sample.ndim == 1:
                    sample = sample[:, None]  # shape (num_envs, 1) for discrete
                # logprob is (num_envs,) for both discrete and continuous distributions
                logprob = logprob[:, None]   # make it (num_envs, 1) for stacking
                actions_list.append(sample)
                log_probs_list.append(logprob)
            # Stack results: actions shape (num_drones, num_envs, action_dim), log_probs shape (num_drones, num_envs, 1)
            actions_arr = jnp.array(actions_list)            # shape (num_drones, num_envs, action_dim)
            log_probs_arr = jnp.array(log_probs_list)        # shape (num_drones, num_envs, 1)
            # Reorder to (num_envs, num_drones, action_dim) for env step, and (num_envs, num_drones) for log_probs storage
            joint_actions = actions_arr.transpose((1, 0, 2))      # shape (num_envs, num_drones, action_dim)
            log_probs = log_probs_arr.squeeze(-1).transpose((1, 0))  # shape (num_envs, num_drones)
            return joint_actions, log_probs

        # Vectorized critic value function for all envs
        vmapped_value = vmap(critic.apply, in_axes=(None, 0))  # maps critic over batch of global_obs

        # Step the environment and collect a transition
        def _env_step(runner_state, _):
            actor_st, critic_st, obs, env_st, rng = runner_state
            last_obs = obs  # save current observation (to store in transition)
            # Policy: get actions for current obs
            rng, subkey = jax.random.split(rng)
            pi_list = _ma_get_pi(actor_st.params, obs)
            joint_actions, log_probs = _ma_sample_and_log_prob_from_pi(pi_list, subkey)
            # Critic: evaluate value for current state
            global_obss = env.state(env_st)            # shape (num_envs, global_state_dim)
            values = vmapped_value(critic_st.params, global_obss)  # shape (num_envs,)
            # Step the environment
            rng, subkey = jax.random.split(rng)
            step_keys = jax.random.split(subkey, args.num_envs)
            obs, rewards, terms, truncs, info, env_st = env.step(env_st, joint_actions, jnp.stack(step_keys))
            # Compute team reward and done flags for each env
            reward = rewards.sum(axis=-1)  # sum over agents (shape: num_envs,)
            done = jnp.logical_or(jnp.any(terms, axis=-1), jnp.any(truncs, axis=-1))  # shape: (num_envs,)
            transition = Transition(
                terminated=done,
                joint_actions=joint_actions,
                value=values,
                reward=reward,
                log_prob=log_probs,
                obs=last_obs,
                global_obs=global_obss,
                info=info,
            )
            return (actor_st, critic_st, obs, env_st, rng), transition

        # Calculate GAE advantages and targets given a trajectory batch and final value
        def _calculate_gae(traj_batch, last_val):
            # Extract arrays for done flags, values, rewards from the trajectory
            dones   = traj_batch.terminated  # shape (T, num_envs)
            values  = traj_batch.value       # shape (T, num_envs)
            rewards = traj_batch.reward      # shape (T, num_envs)

            def _gae_step(carry, x):
                gae, next_val = carry
                done, value, reward = x  # x is a tuple of (done, value, reward) arrays
                delta = reward + args.gamma * next_val * (1 - done) - value
                new_gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                return (new_gae, value), new_gae

            # Scan over the arrays. Here we pass a tuple (dones, values, rewards) as xs; 
            # JAX will iterate over each array along the time axis simultaneously.
            _, advantages_rev = jax.lax.scan(
                _gae_step,
                (jnp.zeros_like(last_val), last_val),
                (dones, values, rewards),
                reverse=True
            )
            advantages = advantages_rev[::-1]
            targets = advantages + values
            return advantages, targets

        # PPO update step for one minibatch
        def _update_minbatch(train_states, batch_data):
            (actor_st, critic_st) = train_states
            (traj_batch, adv_batch, target_batch) = batch_data
            # Define loss for actor and critic
            def loss_fn(params_actor, params_critic, transition: Transition, advantages, targets):
                # Flatten batch and agent dims for batched call
                B, A, D = transition.obs.shape
                obs_flat = transition.obs.reshape((B * A, D))  # (B*A, D)
                act_flat = transition.joint_actions.reshape((B * A, -1))  # (B*A, act_dim)

                # Forward pass
                dists = actor.apply(params_actor, obs_flat)  # batched distribution
                log_probs_new = dists.log_prob(act_flat).reshape((B, A))  # (B, A)
                old_log_probs = transition.log_prob  # (B, A)

                # Ratio and normalized advantage
                log_ratio = log_probs_new.sum(axis=-1) - old_log_probs.sum(axis=-1)
                ratio = jnp.exp(log_ratio)
                adv_mean = advantages.mean()
                adv_std = advantages.std() + 1e-8
                normed_adv = (advantages - adv_mean) / adv_std

                # PPO loss
                pg_loss1 = -normed_adv * ratio
                pg_loss2 = -normed_adv * jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps)
                policy_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                # Critic loss
                values_pred = critic.apply(params_critic, transition.global_obs)
                value_loss = ((values_pred - targets) ** 2).mean()

                # Entropy (mean across batch)
                entropy = dists.entropy().reshape((B, A)).sum(axis=-1).mean()

                # Total loss
                total_loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy
                return total_loss, (policy_loss, value_loss, entropy, log_ratio.mean())

            # Compute gradients
            (loss_val, (pg_loss, v_loss, entropy, kl)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
                actor_st.params, critic_st.params, traj_batch, adv_batch, target_batch
            )
            # Clip gradient norm
            grads = jax.tree_util.tree_map(lambda g: jnp.nan_to_num(g, 0.0), grads)  # sanitize NaNs
            # Update actor and critic train states
            new_actor_state = actor_st.apply_gradients(grads=grads[0])
            new_critic_state = critic_st.apply_gradients(grads=grads[1])
            return (new_actor_state, new_critic_state), (loss_val, pg_loss, v_loss, entropy, kl)

        # Run one full environment rollout (num_steps)
        def _rollout(runner_state, _):
            actor_st, critic_st, obs, env_st, rng = runner_state
            # Collect trajectory of length num_steps using scan
            runner_state, traj_batch = jax.lax.scan(_env_step, (actor_st, critic_st, obs, env_st, rng), None, length=args.num_steps)
            # traj_batch is a batch of Transition of length num_steps (with shapes [num_steps, ...])
            # Compute advantages and targets
            _, _, last_obs, env_st, rng = runner_state
            global_obss = env.state(env_st)  # global obs after final step
            last_values = critic.apply(critic_st.params, global_obss)  # shape (num_envs,)
            advantages, targets = _calculate_gae(traj_batch, last_values)
            # Return the state and the collected data
            return (actor_st, critic_st, last_obs, env_st, rng), (traj_batch, advantages, targets)

        # Perform one PPO update (multiple epochs over minibatches) after collecting trajectories
        def _update_epoch(train_state, _):
            actor_st, critic_st, traj_batch, advantages, targets, rng = train_state
            # Flatten trajectories across time and environments for shuffling
            batch_size = args.num_steps * args.num_envs
            # reshape traj batch data: combine step and env dimensions
            traj_flat = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,)+x.shape[2:]), traj_batch)
            adv_flat = advantages.reshape((batch_size,))
            target_flat = targets.reshape((batch_size,))
            # Shuffle indices
            rng, perm_key = jax.random.split(rng)
            permutation = jax.random.permutation(perm_key, batch_size)
            # Shuffle the flattened batch
            traj_shuf = jax.tree_util.tree_map(lambda x: x[permutation], traj_flat)
            adv_shuf = adv_flat[permutation]
            target_shuf = target_flat[permutation]
            # Split into minibatches
            batch_splits = args.num_minibatches
            traj_mb = jax.tree_util.tree_map(lambda x: x.reshape((batch_splits, -1) + x.shape[1:]), traj_shuf)
            adv_mb = adv_shuf.reshape((batch_splits, -1))
            tgt_mb = target_shuf.reshape((batch_splits, -1))
            # Scan over minibatches to update actor & critic
            (actor_st, critic_st), loss_info = jax.lax.scan(_update_minbatch, (actor_st, critic_st), (traj_mb, adv_mb, tgt_mb), length=batch_splits)
            return (actor_st, critic_st, traj_batch, advantages, targets, rng), loss_info

        # Combine rollout and update into one training iteration
        def _train_iter(runner_state, _):
            # Rollout trajectories
            runner_state, traj_data = _rollout(runner_state, None)
            traj_batch, advantages, targets = traj_data
            # Update policy for multiple epochs
            update_state = (runner_state[0], runner_state[1], traj_batch, advantages, targets, runner_state[4])
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, length=args.update_epochs)
            # After update, carry forward updated actor/critic and current env state & obs
            new_runner_state = (update_state[0], update_state[1], runner_state[2], runner_state[3], update_state[5])
            # We can optionally collect metrics here (like loss_info) if needed
            return new_runner_state, loss_info

        # Initialize runner state (actor_state, critic_state, obs, env_states, key)
        runner_state = (actor_state, critic_state, obs, env_states, key)
        # Run training iterations
        for update in tqdm(range(int(num_updates)), desc="Training Updates"):
            runner_state, loss_info = _train_iter(runner_state, None)
            if args.debug:
                # Print debug info for the last training iteration's losses
                loss_vals = jax.tree_util.tree_map(lambda x: np.array(x).mean(), loss_info)
                # loss_info contains per-minibatch info for each epoch. We take the mean over all data.
                total_loss = loss_vals[0].mean() if isinstance(loss_vals[0], np.ndarray) else loss_vals[0]
                policy_loss = loss_vals[1].mean() if isinstance(loss_vals[1], np.ndarray) else loss_vals[1]
                value_loss = loss_vals[2].mean() if isinstance(loss_vals[2], np.ndarray) else loss_vals[2]
                entropy = loss_vals[3].mean() if isinstance(loss_vals[3], np.ndarray) else loss_vals[3]
                kl = loss_vals[4].mean() if isinstance(loss_vals[4], np.ndarray) else loss_vals[4]
                print(f"[Update {update+1}/{int(num_updates)}] total_loss={float(total_loss):.3f}, "
                      f"policy_loss={float(policy_loss):.3f}, value_loss={float(value_loss):.3f}, "
                      f"entropy={float(entropy):.3f}, approx_kl={float(kl):.3f}")
        return {"runner_state": runner_state}
    return train

if __name__ == "__main__":
    args = parse_args()
    rng = jax.random.PRNGKey(args.seed)
    start_time = time.time()
    # Compile and run training
    print("Number of parallel environments:", args.num_envs)
    train_fn = make_train(args)
    # You can run without JIT for easier debugging:
    if args.debug:
        out = train_fn(rng, None)
    else:
        train_jit = jax.jit(train_fn)
        out = jax.block_until_ready(train_jit(rng, None))
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")
