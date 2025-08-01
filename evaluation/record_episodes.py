import torch
import gymnasium as gym
import numpy as np
import pygame
import imageio
import time
import os

from environment.custom_env import HealthInsuranceEnv
from training.reinforce_training import PolicyNetwork

# Load trained REINFORCE model
model_path = "models/pg/reinforce_insurewise.pt"
env = HealthInsuranceEnv(render_mode="rgb_array")
obs, _ = env.reset()
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

policy_net = PolicyNetwork(input_dim, output_dim)
policy_net.load_state_dict(torch.load(model_path))
policy_net.eval()

# Simulation parameters
EPISODES = 3
GIF_PATH = "evaluation/reinforce_simulation.gif"
frames = []

font = pygame.font.SysFont('arial', 18, bold=True)

def get_overlay_text(info, reward, done, step):
    overlay = f"Step {step} | Profile: {info.get('profile', 'N/A')} | Reward: {reward:.1f}"
    if done:
        if reward >= 15:
            overlay += " | Terminated: ✅ Success"
        else:
            overlay += " | Terminated: ❌ Penalty"
    return overlay

print("🎬 Recording Episodes...")
for ep in range(EPISODES):
    print(f"🎬 Episode {ep+1}")
    obs, info = env.reset()
    done = False
    ep_reward = 0
    step = 0

    while not done:
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            probs = policy_net(obs_tensor)
        action = torch.multinomial(probs, 1).item()

        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        step += 1

        # Render frame and overlay text
        frame = env.render()
        surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        text = get_overlay_text(info, reward, done, step)

        overlay = pygame.Surface((env.renderer.window_size, 30))
        overlay.fill((255, 255, 255))
        label = font.render(text, True, (0, 0, 0))
        overlay.blit(label, (10, 5))

        surface.blit(overlay, (0, 0))
        final_frame = pygame.surfarray.array3d(surface)
        final_frame = np.transpose(final_frame, (1, 0, 2))
        frames.append(final_frame)

        time.sleep(0.1)  # Slow down to visualize

    print(f"Episode {ep+1} Reward: {ep_reward:.2f}")

# Save final GIF
print(f"Saving to {GIF_PATH}")
imageio.mimsave(GIF_PATH, frames, fps=3)
env.close()
print("Simulation complete.")
