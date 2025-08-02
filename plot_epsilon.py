# plot_epsilon.py

import pickle
import matplotlib.pyplot as plt

# Load epsilon data
with open("logs/dqn/epsilon_log.pkl", "rb") as f:
    data = pickle.load(f)

timesteps = data["steps"]
epsilons = data["epsilon"]

# Plot
plt.figure(figsize=(8, 5))
plt.plot(timesteps, epsilons, color='blue', linewidth=2)
plt.title("Epsilon Decay Over Training Steps")
plt.xlabel("Timesteps")
plt.ylabel("Exploration Rate (ε)")
plt.grid(True)
plt.tight_layout()
plt.savefig("epsilon_decay_plot.png")
plt.show()
