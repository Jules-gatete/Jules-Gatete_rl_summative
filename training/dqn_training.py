# training/dqn_training.py

import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from environment.custom_env import HealthInsuranceEnv

class EpsilonLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.epsilons = []

    def _on_step(self) -> bool:
        epsilon = self.model.exploration_rate
        self.logger.record("rollout/epsilon", epsilon)
        self.epsilons.append(epsilon)
        return True

def train_dqn():
    env = Monitor(HealthInsuranceEnv(render_mode="none"))
    log_path = "logs/dqn/"
    os.makedirs(log_path, exist_ok=True)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=500,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        exploration_fraction=0.3,
        verbose=1,
        tensorboard_log=log_path,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path="models/dqn/",
        log_path=log_path,
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1,
    )

    epsilon_logger = EpsilonLogger()
    model.learn(total_timesteps=50000, callback=[eval_callback, epsilon_logger])
    model.save("models/dqn/dqn_model")

    print(" DQN training complete.")

if __name__ == "__main__":
    train_dqn()
