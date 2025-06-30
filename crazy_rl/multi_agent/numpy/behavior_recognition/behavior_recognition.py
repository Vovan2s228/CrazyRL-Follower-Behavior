"""
Behavior-Recognition environment (NumPy version)

Observer outputs label:
  0 = hover, 1 = land, 2 = move
Reward: +1 correct, −0.1 otherwise.
"""

from typing import Optional           # <-- NEW
import numpy as np
from gymnasium import spaces
from crazy_rl.multi_agent.numpy.base_parallel_env import BaseParallelEnv


class BehaviorRecognition(BaseParallelEnv):
    metadata = {"render_modes": ["human", "real"], "is_parallelizable": True, "render_fps": 20}

    def __init__(
        self,
        drone_ids=np.array([0]),
        init_flying_pos=np.array([[0.0, 0.0, 1.0]]),
        init_peer_pos=np.array([1.0, 1.0, 1.5]),
        size: int = 2,
        render_mode=None,
        peer_behavior: Optional[str] = None,    
        target_id: Optional[str] = None,
        swarm=None,
    ):
        self.size = size
        self.render_mode = render_mode
        self.swarm = swarm
        self.drone_ids = drone_ids
        self.target_id = target_id

        self._mode = "real" if self.render_mode == "real" else "human"

        self._agents_names = np.array([f"agent_{i}" for i in drone_ids])
        self.possible_agents = self._agents_names.tolist()
        self.agents = []
        self.timestep = 0
        self.max_steps = 50

        self._init_flying_pos = {self._agents_names[0]: init_flying_pos[0].astype(np.float32)}
        self._init_peer_pos = init_peer_pos.astype(np.float32)

        self._agent_location = {}
        self._peer_location = None
        self._peer_velocity = None
        self.behavior_label = None
        self.last_guess = None

        self._preset_behavior = peer_behavior

    # ------------------------------------------------------------------
    def _observation_space(self, agent):
        pos_low = np.array([-self.size, -self.size, 0], dtype=np.float32)
        pos_high = np.array([self.size, self.size, 3], dtype=np.float32)
        vel_low = np.array([-3.0, -3.0, -3.0], dtype=np.float32)
        vel_high = np.array([3.0, 3.0, 3.0], dtype=np.float32)
        low = np.concatenate([pos_low, pos_low, vel_low])
        high = np.concatenate([pos_high, pos_high, vel_high])
        return spaces.Box(low=low, high=high, shape=(9,), dtype=np.float32)

    def _action_space(self, agent):
        return spaces.Discrete(3)

    # ------------------------------------------------------------------
    def _compute_obs(self):
        agent = self._agents_names[0]
        obs_vec = np.concatenate(
            [self._agent_location[agent], self._peer_location, self._peer_velocity]
        ).astype(np.float32)
        return {agent: obs_vec}

    def _transition_state(self, actions):
        agent = self._agents_names[0]
        self.last_guess = int(actions[agent])

        if self.behavior_label == 0:
            self._peer_velocity = np.zeros(3, dtype=np.float32)
        elif self.behavior_label == 1:
            self._peer_velocity = np.array([0, 0, -0.1], dtype=np.float32)
        elif self.behavior_label == 2:
            if not hasattr(self, "_move_dir"):
                v = np.random.randn(2); v /= np.linalg.norm(v)
                self._move_dir = np.array([v[0], v[1], 0], dtype=np.float32)
            self._peer_velocity = 0.1 * self._move_dir

        self._peer_location = np.clip(
            self._peer_location + self._peer_velocity,
            [-self.size, -self.size, 0.0],
            [self.size, self.size, 3.0],
        )
        return {agent: self._agent_location[agent]}

    def _compute_reward(self):
        agent = self._agents_names[0]
        return {agent: 1.0 if self.last_guess == self.behavior_label else -0.1}

    def _compute_terminated(self):
        agent = self._agents_names[0]
        return {agent: self.last_guess == self.behavior_label}

    def _compute_truncation(self):
        agent = self._agents_names[0]
        return {agent: self.timestep >= self.max_steps}

    def _compute_info(self):
        return {self._agents_names[0]: {"true_label": self.behavior_label}}

    # ------------------------------------------------------------------
    def reset(self, seed=None, return_info=False, options=None):
        # super().reset(...) removed because BaseParallelEnv expects target vars
        self.timestep = 0
        self.agents = self.possible_agents.copy()

        self._agent_location = self._init_flying_pos.copy()
        self._peer_location = self._init_peer_pos.copy()
        self._peer_velocity = np.zeros(3, dtype=np.float32)

        if self._preset_behavior is None:
            self.behavior_label = np.random.randint(0, 3)
        else:
            self.behavior_label = {"hover": 0, "land": 1, "move": 2}[self._preset_behavior]

        if hasattr(self, "_move_dir"):
            delattr(self, "_move_dir")

        obs = self._compute_obs()
        info = self._compute_info()
        return (obs, info) if return_info else obs
