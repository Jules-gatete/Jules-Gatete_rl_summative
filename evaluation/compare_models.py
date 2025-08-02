# evaluation/compare_models.py

import numpy as np
import torch
from stable_baselines3 import DQN, PPO, A2C
from environment.custom_env import HealthInsuranceEnv
from training.reinforce_training import PolicyNetwork

def evaluate_model(agent_type, model_path, episodes=5):
    env = HealthInsuranceEnv(render_mode="none")
    rewards = []

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
        raise ValueError("Invalid agent type")

    for ep in range(episodes):
        obs, _ = env.reset()
        done, truncated = False, False
        total_reward = 0

        while not (done or truncated):
            if agent_type == "reinforce":
                with torch.no_grad():
                    probs = model(torch.tensor(obs, dtype=torch.float32))
                    action = torch.argmax(probs).item()
            else:
                action, _ = model.predict(obs)

            obs, reward, done, truncated, _ = env.step(action)
            total_reward += reward

        rewards.append(total_reward)
        print(f" {agent_type.upper()} Episode {ep+1} Reward: {total_reward:.2f}")

    avg = np.mean(rewards)
    print(f"\n {agent_type.upper()} Average Reward over {episodes} episodes: {avg:.2f}")
    return avg

if __name__ == "__main__":
    models = {
     "dqn": "models/dqn/dqn_model.zip",
    "ppo": "models/pg/ppo_insurewise.zip",
    "a2c": "models/pg/a2c_insurewise.zip",
    "reinforce": "models/pg/reinforce_insurewise.pt"
    }

    for name, path in models.items():
        print(f"\n Evaluating {name.upper()}...")
        evaluate_model(name, path)
