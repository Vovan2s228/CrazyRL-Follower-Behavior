# crazy_rl/multi_agent/jax/behavior_recognition/behavior_recognition.py
"""
JAX functional version of the Behavior-Recognition environment.

One learning agent (“observer”) predicts the peer drone’s behaviour:
    0 = hover,  1 = land,  2 = move

Action space:     Discrete(3)
Observation size: 9  (observer pos 3  | peer pos 3  | peer vel 3)
Reward: +1 correct, −0.1 wrong
Episode terminates on correct guess or after max_steps.
"""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import jit, random

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State
from crazy_rl.utils.jax_spaces import Box, Discrete, Space
from crazy_rl.utils.jax_wrappers import AutoReset, VecEnv


@jdc.pytree_dataclass
class BRState(State):
    """Complete environment state (pytree)."""
    agents_locations: jnp.ndarray       # (num_agents, 3)
    timestep: int
    peer_location: jnp.ndarray          # (3,)
    peer_velocity: jnp.ndarray          # (3,)
    behavior_label: int                 # 0|1|2
    random_dir: jnp.ndarray          # (2,) horizontal direction for “move”


class BehaviorRecognitionJax(BaseParallelEnv):
    """Stateless functional env compatible with CrazyRL JAX MAPPO."""

    def __init__(
        self,
        init_agent_pos: jnp.ndarray,
        init_peer_pos: jnp.ndarray,
        size: float = 2.0,
        max_steps: int = 50,
    ):
        self.num_drones = init_agent_pos.shape[0]       # usually 1
        self.size = size
        self.max_steps = max_steps
        self._init_agent_pos = init_agent_pos
        self._init_peer_pos = init_peer_pos

    # ------------------------------------------------------------------ #
    # Spaces
    # ------------------------------------------------------------------ #
    def observation_space(self, agent: int) -> Space:
        pos_low = jnp.array([-self.size, -self.size, 0.0])
        pos_high = jnp.array([self.size, self.size, 3.0])
        vel_low = jnp.array([-3.0, -3.0, -3.0])
        vel_high = jnp.array([3.0, 3.0, 3.0])
        low = jnp.concatenate([pos_low, pos_low, vel_low])
        high = jnp.concatenate([pos_high, pos_high, vel_high])
        return Box(low, high, (9,))

    def action_space(self, agent: int) -> Space:
        return Discrete(3)                              # 0 / 1 / 2

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state: BRState) -> jnp.ndarray:
        """Return per-agent observation array, shape (num_agents, 9)."""
        obs = jnp.concatenate(
            [state.agents_locations[0], state.peer_location, state.peer_velocity]
        )
        return obs[None, :]  # (1, 9)

    # scripted peer motion
    # scripted peer motion
    @partial(jit, static_argnums=(0,))
    def _transition_state(
        self,
        state: BRState,
        actions: jnp.ndarray,
        key: jnp.ndarray,
    ) -> Tuple[BRState, int]:
        guess = actions[0]          # scalar int
        beh   = state.behavior_label

        # peer velocity per behaviour
        vel_hover = jnp.zeros(3)
        vel_land  = jnp.array([0.0, 0.0, -0.1])
        vel_move  = jnp.concatenate([state.random_dir * 0.1, jnp.array([0.0])])

        # pick the correct one by behavior_label
        vels = jnp.stack([vel_hover, vel_land, vel_move])
        vel  = vels[beh]

        peer_loc = jnp.clip(
            state.peer_location + vel,
            jnp.array([-self.size, -self.size, 0.0]),
            jnp.array([ self.size,  self.size, 3.0]),
        )

        state = jdc.replace(
            state,
            timestep      = state.timestep + 1,
            peer_location = peer_loc,
            peer_velocity = vel,
        )
        return state, guess


    # ------------------------------------------------------------------ #
    # Public API (reset, step, state)
    # ------------------------------------------------------------------ #
    @partial(jit, static_argnums=(0,))
    def reset(self, key: jnp.ndarray):
        key, sub = random.split(key)
        behavior   = random.randint(sub, (), 0, 3)
        key, sub = random.split(key)
        random_dir = random.uniform(sub, (2,), minval=-1.0, maxval=1.0)
        random_dir = random_dir / (jnp.linalg.norm(random_dir) + 1e-6)

        state = BRState(
            agents_locations=self._init_agent_pos,
            timestep=0,
            peer_location=self._init_peer_pos,
            peer_velocity=jnp.zeros(3),
            behavior_label=behavior,
            random_dir=random_dir,
        )
        obs = self._compute_obs(state)
        return obs, {}, state

    @partial(jit, static_argnums=(0,))
    def step(
        self,
        state: BRState,
        actions: jnp.ndarray,
        key: jnp.ndarray,
    ):
        state, guess = self._transition_state(state, actions, key)
        correct = jnp.equal(guess, state.behavior_label)
        reward = jnp.where(correct, 1.0, -0.1)[None]
        terminated = jnp.where(correct, 1, 0)[None]
        truncated = jnp.where(state.timestep >= self.max_steps, 1, 0)[None]
        obs = self._compute_obs(state)
        return obs, reward, terminated, truncated, {}, state

    @partial(jit, static_argnums=(0,))
    def state(self, state: BRState) -> jnp.ndarray:
        return jnp.concatenate(
            [state.agents_locations.flatten(), state.peer_location, state.peer_velocity]
        )
