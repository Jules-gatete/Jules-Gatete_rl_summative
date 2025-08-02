import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from environment.custom_env import HealthInsuranceEnv
from gymnasium.spaces import flatten
from gymnasium.spaces.utils import flatten_space
from torch.utils.tensorboard import SummaryWriter

# Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)

#  Discount rewards
def compute_returns(rewards, gamma=0.99):
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)

# Training
def train_reinforce(env, policy, optimizer, episodes=1000, gamma=0.99, log_dir="logs/pg/reinforce/"):
    writer = SummaryWriter(log_dir)
    all_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        obs = torch.tensor(flatten(env.observation_space, obs), dtype=torch.float32)
        done = False

        log_probs = []
        rewards = []
        total_reward = 0

        while not done:
            probs = policy(obs)
            dist = Categorical(probs)
            action = dist.sample()

            log_probs.append(dist.log_prob(action))
            obs, reward, terminated, truncated, _ = env.step(action.item())
            obs = torch.tensor(flatten(env.observation_space, obs), dtype=torch.float32)

            done = terminated or truncated
            rewards.append(reward)
            total_reward += reward

        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        loss = -torch.sum(torch.stack(log_probs) * returns)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        all_rewards.append(total_reward)
        writer.add_scalar("Reward", total_reward, ep)
        writer.add_scalar("Loss", loss.item(), ep)

        if (ep + 1) % 50 == 0:
            avg = np.mean(all_rewards[-50:])
            print(f"Episode {ep + 1} | Avg Reward (last 50): {round(avg, 2)}")

    writer.close()
    return policy

#  Entry
if __name__ == "__main__":
    env = HealthInsuranceEnv()
    input_dim = flatten_space(env.observation_space).shape[0]
    output_dim = env.action_space.n

    policy = PolicyNetwork(input_dim, output_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.001)

    trained_policy = train_reinforce(env, policy, optimizer, episodes=1000)

    os.makedirs("models/pg", exist_ok=True)
    torch.save(trained_policy.state_dict(), "models/pg/reinforce_insurewise.pt")
    print(" REINFORCE training complete. Model saved to models/pg/reinforce_insurewise.pt")
