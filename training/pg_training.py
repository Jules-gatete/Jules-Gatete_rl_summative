import argparse
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_env import HealthInsuranceEnv

parser = argparse.ArgumentParser()
parser.add_argument("--algo", choices=["ppo", "a2c"], required=True, help="Algorithm to train")
args = parser.parse_args()

log_dir = f"logs/pg/{args.algo}/"
model_path = f"models/pg/{args.algo}_insurewise.zip"
os.makedirs(log_dir, exist_ok=True)

env = make_vec_env(lambda: Monitor(HealthInsuranceEnv()), n_envs=1)

if args.algo == "ppo":
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
elif args.algo == "a2c":
    model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

eval_callback = EvalCallback(env, best_model_save_path=f"models/pg/{args.algo}_best/",
                             log_path=log_dir, eval_freq=5000,
                             deterministic=True, render=False,
                             callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=18.0, verbose=1))

model.learn(total_timesteps=100_000, callback=eval_callback)
model.save(model_path)

print(f" {args.algo.upper()} training complete. Model saved to {model_path}")
