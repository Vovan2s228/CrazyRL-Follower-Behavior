import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax import random, jit
from functools import partial
from typing import Tuple

from crazy_rl.multi_agent.jax.base_parallel_env import BaseParallelEnv, State
from crazy_rl.utils.jax_spaces import Box, Space


# --------------------------------------------------------------------------- #
#  Pytree state                                                               #
# --------------------------------------------------------------------------- #
@jdc.pytree_dataclass
class FBState(State):
    agents_locations: jnp.ndarray   # (num_drones, 3) follower positions
    timestep: int
    leader_location: jnp.ndarray    # (3,) leader pos
    leader_velocity: jnp.ndarray    # (3,) leader vel
    behavior_label: int             # 0 = hover, 1 = move, 2 = land
    random_dir: jnp.ndarray         # (2,) horiz dir used in move phase
    agents_velocities: jnp.ndarray  # (num_drones, 3) follower vels


# --------------------------------------------------------------------------- #
#  Environment                                                                #
# --------------------------------------------------------------------------- #
class EscortFollowerBehavior(BaseParallelEnv):
    """Follower learns to react to scripted leader (hover / move / land)."""

    # ----------------------------- init ----------------------------------- #
    def __init__(
        self,
        init_flying_pos: jnp.ndarray,
        init_leader_pos: jnp.ndarray,
        size: float = 2.0,
        steps_per_phase: int = 50,
    ):
        self.num_drones = init_flying_pos.shape[0]
        self.size = size
        self.steps_per_phase = steps_per_phase
        self.possible_agents = [f"agent_{i}" for i in range(self.num_drones)]

        self._init_agent_pos = init_flying_pos
        self._init_leader_pos = init_leader_pos
        self.max_steps = 3 * steps_per_phase  # 3 phases

    # --------------------------- spaces ----------------------------------- #
    def observation_space(self, agent: int) -> Space:
        low  = jnp.array([-self.size, -self.size, 0.0] +     # self pos
                         [-3.0, -3.0, -3.0] +                # self vel
                         [-self.size, -self.size, 0.0] +     # leader pos
                         [-3.0, -3.0, -3.0])                 # leader vel
        high = jnp.array([ self.size,  self.size, 3.0] +
                         [ 3.0,  3.0,  3.0] +
                         [ self.size,  self.size, 3.0] +
                         [ 3.0,  3.0,  3.0])
        return Box(low, high, shape=(12,))

    def action_space(self, agent: int) -> Space:
        return Box(low=-1.0, high=1.0, shape=(3,))

    # ----------------------- helper: observation -------------------------- #
    @partial(jit, static_argnums=(0,))
    def _compute_obs(self, state: FBState) -> jnp.ndarray:
        """Return (num_drones, 12) observation tensor."""
        def obs_i(i):
            return jnp.concatenate(
                [
                    state.agents_locations[i],
                    state.agents_velocities[i],
                    state.leader_location,
                    state.leader_velocity,
                ]
            )
        return jax.vmap(obs_i)(jnp.arange(self.num_drones))

    # ----------------------- helper: transition -------------------------- #
    @partial(jit, static_argnums=(0,))
    def _transition_state(
        self, state: FBState, actions: jnp.ndarray, key: jnp.ndarray
    ) -> FBState:
        phase = jnp.clip((state.timestep - 1) // self.steps_per_phase, 0, 2)

        vel_hover = jnp.zeros(3)
        vel_move  = jnp.concatenate([state.random_dir * 0.1, jnp.array([0.0])])
        vel_land  = jnp.array([0.0, 0.0, -0.1])
        leader_vel = jnp.select(
            [phase == 0, phase == 1, phase == 2],
            [vel_hover, vel_move, vel_land],
        )

        leader_loc = jnp.clip(
            state.leader_location + leader_vel,
            jnp.array([-self.size, -self.size, 0.0]),
            jnp.array([ self.size,  self.size, 3.0]),
        )

        new_pos = self._sanitize_action(state, actions)
        follower_vel = new_pos - state.agents_locations

        return jdc.replace(
            state,
            agents_locations=new_pos,
            leader_location=leader_loc,
            leader_velocity=leader_vel,
            behavior_label=phase,
            agents_velocities=follower_vel,
        )

    # ---------------------- reward / done / trunc ------------------------ #
    @partial(jit, static_argnums=(0,))
    def _compute_reward(
        self, state: FBState, *, terminations: jnp.ndarray, truncations: jnp.ndarray
    ) -> jnp.ndarray:
        return -jnp.linalg.norm(state.agents_locations - state.leader_location, axis=1)

    @partial(jit, static_argnums=(0,))
    def _compute_terminated(self, state: FBState) -> jnp.ndarray:
        return jnp.zeros(self.num_drones, dtype=jnp.int32)

    @partial(jit, static_argnums=(0,))
    def _compute_truncation(self, state: FBState) -> jnp.ndarray:
        trunc = state.timestep >= self.max_steps
        return jnp.ones(self.num_drones, dtype=jnp.int32) * jnp.array(trunc, dtype=jnp.int32)

    # ----------------------  reset (wrapper) ----------------------------- #
    def reset(self, key: jnp.ndarray):
        key, subkey = random.split(key)
        rand_dir = random.uniform(subkey, (2,), minval=-1.0, maxval=1.0)
        rand_dir = rand_dir / (jnp.linalg.norm(rand_dir) + 1e-6)

        state = FBState(
            agents_locations=self._init_agent_pos,
            timestep=0,
            leader_location=self._init_leader_pos,
            leader_velocity=jnp.zeros(3),
            behavior_label=0,
            random_dir=rand_dir,
            agents_velocities=jnp.zeros_like(self._init_agent_pos),
        )
        obs = self._compute_obs(state)
        info = {agent: {"true_label": 0} for agent in self.possible_agents}
        return obs, info, state

    # --------------------- step: jitted core + wrapper ------------------- #
    @partial(jit, static_argnums=(0,))
    def _step_core(self, state: FBState, actions: jnp.ndarray, key: jnp.ndarray):
        state = jdc.replace(state, timestep=state.timestep + 1)
        state = self._transition_state(state, actions, key)

        terminated = self._compute_terminated(state)
        truncated  = self._compute_truncation(state)
        rewards    = self._compute_reward(state, terminations=terminated, truncations=truncated)
        obs        = self._compute_obs(state)
        return obs, rewards, terminated, truncated, state

    # (not jitted) – builds Python dict, so no tracer→Python conversion
    def step(self, state: FBState, actions: jnp.ndarray, key: jnp.ndarray):
        obs, rewards, terminated, truncated, state = self._step_core(state, actions, key)
        label_scalar = int(state.behavior_label.item())      # now concrete
        info = {agent: {"true_label": label_scalar} for agent in self.possible_agents}
        return obs, rewards, terminated, truncated, info, state
