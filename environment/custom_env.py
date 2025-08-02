import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional
import pygame
import os

class HealthInsuranceEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        self.rows, self.cols = 4, 4
        self.render_mode = render_mode
        self.window_size = 512
        self.max_steps = 50
        self.steps = 0

        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

        self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        self._load_assets()

    def _load_assets(self):
        pygame.font.init()
        self.emoji_font = pygame.font.SysFont("Segoe UI Emoji", 40)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.visited_health = False
        self.visited_income = False
        self.reached_target = False
        self.last_reward = 0
        self.correct_plan = None

        self.agent_pos = [3, 3]
        self.health_pos = [1, 2]
        self.income_pos = [1, 0]
        self.target_pos = [0, 0]
        self.misinfo_pos = [0, 3]
        self.barrier_pos = [3, 0]

        self.health_profile = self.np_random.choice(["chronic", "maternity", "disabled", "general"])
        self.income_level = self.np_random.choice(["low", "medium", "high"])

        if self.health_profile == "chronic" and self.income_level == "low":
            self.correct_plan = "CBHI"
        elif self.health_profile == "maternity" and self.income_level == "medium":
            self.correct_plan = "RAMA"
        elif self.health_profile == "disabled" and self.income_level == "high":
            self.correct_plan = "MMI"
        else:
            self.correct_plan = "Private"

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.agent_pos[0] / self.rows,
            self.agent_pos[1] / self.cols,
            float(self.visited_health),
            float(self.visited_income),
            int(self.health_profile == "chronic"),
            int(self.health_profile == "maternity"),
            int(self.health_profile == "disabled"),
            int(self.income_level == "low"),
            int(self.income_level == "medium"),
            int(self.income_level == "high")
        ], dtype=np.float32)

    def step(self, action):
        self.steps += 1
        direction = self._action_to_direction[int(action)]
        new_pos = [self.agent_pos[0] + direction[0], self.agent_pos[1] + direction[1]]

        if 0 <= new_pos[0] < self.rows and 0 <= new_pos[1] < self.cols:
            self.agent_pos = new_pos

        reward = -0.1
        terminated = False
        truncated = False

        if self.agent_pos == self.health_pos:
            self.visited_health = True
            reward += 0.5

        if self.agent_pos == self.income_pos:
            self.visited_income = True
            reward += 0.5

        if self.agent_pos == self.misinfo_pos:
            reward = -5
            terminated = True

        if self.agent_pos == self.barrier_pos:
            reward = -5
            terminated = True

        if self.agent_pos == self.target_pos:
            self.reached_target = True
            if self.visited_health and self.visited_income:
                reward = 10
            else:
                reward = -3
            terminated = True

        if self.steps >= self.max_steps:
            truncated = True

        self.last_reward = reward
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        from environment.rendering import HealthInsuranceRenderer
        if not hasattr(self, 'renderer'):
            self.renderer = HealthInsuranceRenderer(self, render_mode=self.render_mode)
        return self.renderer.render()

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()
