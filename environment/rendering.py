# environment/rendering.py
import pygame
import numpy as np
from typing import Optional
import os

class HealthInsuranceRenderer:
    def __init__(self, env, render_mode: Optional[str] = None):
        self.env = env
        self.render_mode = render_mode
        self.window_size = 512
        self.status_height = 100
        self.total_height = self.window_size + self.status_height
        self.window = None
        self.clock = None

        pygame.font.init()
        self.font = pygame.font.SysFont("Arial", 24)
        self.small_font = pygame.font.SysFont("Arial", 20)

        self.tile_size = self.window_size // self.env.cols
        self.images = self._load_assets()

    def _load_assets(self):
        def load_img(filename):
            path = os.path.join("assets", filename)
            image = pygame.image.load(path)
            return pygame.transform.scale(image, (self.tile_size - 10, self.tile_size - 10))

        return {
            "agent": load_img("agent.jpg"),
            "health": load_img("health.png"),
            "income": load_img("income.png"),
            "target": load_img("target.png"),
            "misinfo": load_img("misinfo.png"),
            "barrier": load_img("barrier.png"),
        }

    def render(self):
        if self.render_mode == "human":
            return self._render_human()
        elif self.render_mode == "rgb_array":
            return self._render_rgb_array()

    def _render_human(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.total_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.window_size, self.total_height))
        surface.fill((255, 255, 255))

        for row in range(self.env.rows):
            for col in range(self.env.cols):
                rect = pygame.Rect(col * self.tile_size, row * self.tile_size, self.tile_size, self.tile_size)
                pygame.draw.rect(surface, (230, 230, 230), rect, 1)

        def blit(icon_key, pos):
            x, y = pos[1] * self.tile_size + 5, pos[0] * self.tile_size + 5
            surface.blit(self.images[icon_key], (x, y))

        # Target icon (switches after agent reaches it)
        if getattr(self.env, "reached_target", False):
            plan_icons = {
                "RAMA": "RAMA.png",
                "CBHI": "Mutuelle-Card.jpg",
                "MMI": "MMI.png",
                "Private": "Private.png"
            }
            recommended_plan = getattr(self.env, "correct_plan", None)
            if recommended_plan and recommended_plan in plan_icons:
                icon_path = os.path.join("assets", plan_icons[recommended_plan])
                if os.path.exists(icon_path):
                    plan_img = pygame.image.load(icon_path)
                    plan_img = pygame.transform.scale(plan_img, (self.tile_size - 10, self.tile_size - 10))
                    x, y = self.env.target_pos[1] * self.tile_size + 5, self.env.target_pos[0] * self.tile_size + 5
                    surface.blit(plan_img, (x, y))
                else:
                    blit("target", self.env.target_pos)
            else:
                blit("target", self.env.target_pos)
        else:
            blit("target", self.env.target_pos)

        blit("health", self.env.health_pos)
        blit("income", self.env.income_pos)
        blit("misinfo", self.env.misinfo_pos)
        blit("barrier", self.env.barrier_pos)
        blit("agent", self.env.agent_pos)

        pygame.draw.line(surface, (100, 100, 100), (0, self.window_size), (self.window_size, self.window_size), 3)

        profile = f"Health: {self.env.health_profile}, Income: {self.env.income_level}"
        reward = f"Reward: {round(self.env.last_reward, 2)}"
        profile_surf = self.font.render(profile, True, (0, 0, 0))
        reward_color = (0, 120, 0) if self.env.last_reward >= 0 else (200, 0, 0)
        reward_surf = self.font.render(reward, True, reward_color)

        surface.blit(profile_surf, (10, self.window_size + 10))
        surface.blit(reward_surf, (10, self.window_size + 50))

        self.window.blit(surface, surface.get_rect())
        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.env.metadata["render_fps"])

    def _render_rgb_array(self):
        surface = pygame.Surface((self.window_size, self.total_height))
        surface.fill((255, 255, 255))
        return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def close(self):
        if self.window:
            pygame.display.quit()
            pygame.quit()
