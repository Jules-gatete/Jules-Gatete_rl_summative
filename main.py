import argparse
import os
import numpy as np
import imageio
import torch
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import HealthInsuranceEnv
from training.reinforce_training import PolicyNetwork  # 👈 Needed for REINFORCE

def run_random_agent(episodes=3, render_mode="human", save_gif=False):
    env = HealthInsuranceEnv(render_mode=render_mode)
    frames = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render_mode == "rgb_array" and save_gif:
                frame = env.render()
                frames.append(frame)
            elif render_mode == "human":
                env.render()

            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {ep + 1} | Total Reward: {round(total_reward, 2)}")

    if save_gif and frames:
        os.makedirs("gifs", exist_ok=True)
        gif_path = f"gifs/random_agent.gif"
        imageio.mimsave(gif_path, frames, fps=env.metadata["render_fps"])
        print(f"🎞️ Saved random agent GIF at: {gif_path}")

    env.close()

def evaluate_model(algo, model_path, episodes=3, render_mode="human"):
    env = HealthInsuranceEnv(render_mode=render_mode)
    print(f"\n Evaluating {algo.upper()} for {episodes} episodes...")

    # Handle model loading
    if algo == "dqn":
        model = DQN.load(model_path)
        predict_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    elif algo == "ppo":
        model = PPO.load(model_path)
        predict_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    elif algo == "a2c":
        model = A2C.load(model_path)
        predict_fn = lambda obs: model.predict(obs, deterministic=True)[0]
    elif algo == "reinforce":
        obs_space = env.observation_space
        act_space = env.action_space
        obs_sample, _ = env.reset()
        obs_dim = np.array(obs_sample).flatten().shape[0]
        action_dim = act_space.n

        reinforce_model = PolicyNetwork(input_dim=obs_dim, output_dim=action_dim)
        reinforce_model.load_state_dict(torch.load(model_path))
        reinforce_model.eval()

        def predict_fn(obs):
            obs_tensor = torch.tensor(np.array(obs).flatten(), dtype=torch.float32)
            probs = reinforce_model(obs_tensor)
            action = torch.argmax(probs).item()
            return action
    else:
        raise ValueError("Unsupported algorithm")

    total_rewards = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = predict_fn(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if render_mode == "human":
                env.render()
            total_reward += reward

        print(f" Episode {ep + 1} Reward: {round(total_reward, 2)}")
        total_rewards.append(total_reward)

    avg = round(np.mean(total_rewards), 2)
    print(f" Average Reward: {avg}")
    env.close()

def main():
    parser = argparse.ArgumentParser(description="InsureWise RL Environment Runner")
    parser.add_argument("--mode", type=str, choices=["random", "eval"], default="random", help="Mode: random or eval")
    parser.add_argument("--algo", type=str, choices=["dqn", "ppo", "a2c", "reinforce"], help="RL algorithm to evaluate (if mode=eval)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to saved model")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--render", type=str, choices=["human", "rgb_array", "none"], default="human", help="Render mode")
    parser.add_argument("--save_gif", action="store_true", help="Save RGB frames as GIF (only in random + rgb_array mode)")
    args = parser.parse_args()

    if args.mode == "random":
        run_random_agent(episodes=args.episodes, render_mode=args.render, save_gif=args.save_gif)
    elif args.mode == "eval":
        if not args.algo or not args.model_path:
            print("❌ Please provide both --algo and --model_path for evaluation mode.")
        else:
            evaluate_model(algo=args.algo, model_path=args.model_path, episodes=args.episodes, render_mode=args.render)

if __name__ == "__main__":
    main()
