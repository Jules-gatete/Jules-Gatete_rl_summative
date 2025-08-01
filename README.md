# InsureWise: Health Insurance Recommendation via Reinforcement Learning

InsureWise is a reinforcement learning project that simulates a Rwandan citizen navigating a 4×4 grid to collect health and income information before selecting the most suitable insurance plan. The agent must avoid traps and make informed decisions to succeed.

## Objective
Train an agent to:
- Visit health and income stations
- Select the optimal insurance plan (CBHI, RAMA, MMI, Private)
- Avoid traps and barriers
- Maximize cumulative reward

## Environment
- Custom Gymnasium-compatible environment
- Visualized using Pygame
- Rewards for correct steps, penalties for incorrect or premature actions

## Algorithms Compared
- Deep Q-Network (DQN)
- REINFORCE (custom PyTorch implementation)
- Proximal Policy Optimization (PPO)
- Advantage Actor-Critic (A2C)

## Best Performing Model
The REINFORCE model achieved the highest average reward (13.8) across evaluation episodes and demonstrated strong generalization.

## Project Structure
```

InsureWise/
├── assets/
├── environment/
│   ├── custom\_env.py          # Custom environment logic
│   └── rendering.py           # Pygame-based renderer
├── training/
│   ├── dqn\_training.py        # DQN training script
│   ├── pg\_training.py         # PPO and A2C training
│   └── reinforce\_training.py  # Custom REINFORCE training
├── evaluation/
│   ├── compare\_models.py      # Evaluates all trained models
│   └── record\_episodes.py     # Records simulation video
├── models/
│   ├── dqn/
│   └── pg/
├── main.py                    
├── requirements.txt
└── README.md

````

## Example: Evaluate REINFORCE Model
```bash
python main.py --mode eval --algo reinforce --model_path models/pg/reinforce_insurewise.pt --episodes 1 --render human
````

## Installation

```bash
pip install -r requirements.txt
```

## Deliverables

* Source code for environment, training, and evaluation
* Trained models (DQN, PPO, A2C, REINFORCE)
* Video recording of simulation (![alt text](reinforce.gif))

video presentation ![video link](https://youtu.be/kbZqr28gSSE)

