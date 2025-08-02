# evaluation/record_episodes.py

import argparse
import imageio
import os
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import HealthInsuranceEnv
from training.reinforce_training import PolicyNetwork
import torch
import gymnasium as gym
import numpy as np

def record(agent_type, model_path, out_path="episode.gif", render_mode="rgb_array", fps=1):
    env = HealthInsuranceEnv(render_mode=render_mode)
    frames = []

    obs, _ = env.reset()
    done, truncated = False, False

    if agent_type == "dqn":
        model = DQN.load(model_path)
    elif agent_type == "ppo":
        model = PPO.load(model_path)
    elif agent_type == "a2c":
        model = A2C.load(model_path)
    elif agent_type == "reinforce":
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n
        model = PolicyNetwork(input_dim, output_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        raise ValueError("Unsupported agent type.")

    while not (done or truncated):
        if agent_type == "reinforce":
            with torch.no_grad():
                action_probs = model(torch.tensor(obs, dtype=torch.float32))
                action = torch.argmax(action_probs).item()
        else:
            action, _ = model.predict(obs)

        obs, _, done, truncated, _ = env.step(action)
        frame = env.render()
        frames.append(frame)

    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Episode recorded: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["dqn", "ppo", "a2c", "reinforce"], required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--out", default="episode.gif")
    parser.add_argument("--fps", type=int, default=1)
    args = parser.parse_args()

    record(args.agent, args.model_path, args.out, fps=args.fps)
