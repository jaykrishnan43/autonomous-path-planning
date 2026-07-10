# Autonomous Path Planning using Q-Learning

A reinforcement learning agent that learns to navigate a grid-world environment 
from a start point to a goal, avoiding obstacles, using the Q-learning algorithm.

Developed as a major project for course 21CSA699A (MCA).

## Overview

This project implements a tabular Q-learning agent that learns an optimal 
navigation policy through trial-and-error interaction with a grid-world 
environment. The agent has no prior knowledge of the environment and learns 
purely from rewards and penalties received while exploring.

**Core concepts demonstrated:**
- Markov Decision Processes (MDPs)
- The Bellman equation for value updates
- Epsilon-greedy exploration vs. exploitation
- Q-table convergence

## Project Structure

autonomous-path-planning/
├── environment/
│   └── grid_world.py       # Grid environment: states, actions, rewards
├── agent/
│   └── q_learning_agent.py # Q-learning agent: Q-table, action selection, updates
├── training/
│   └── train.py            # Standalone training script
├── visualization/
│   └── visualize.py        # Generates result plots
├── results/                # Saved Q-table, metrics, and plots
├── main.py                 # Main entry point
└── README.md

## How It Works

1. **Environment**: A 10x10 grid with a defined start, goal, and obstacles.
   The agent receives +100 for reaching the goal, -50 for hitting an obstacle,
   -5 for hitting a boundary, and -1 for every other step (encouraging shorter paths).

2. **Agent**: Uses a Q-table to estimate the value of each (state, action) pair.
   Actions are chosen using an epsilon-greedy policy — balancing exploration 
   (random actions) with exploitation (best known action), with epsilon 
   decaying over time.

3. **Learning**: After each step, the Q-table is updated using the Bellman equation:

Q(s,a) ← Q(s,a) + α [r + γ · max(Q(s',a')) − Q(s,a)]

4. **Training**: The agent runs for 1000 episodes, gradually shifting from 
   random exploration to exploiting its learned policy.

## Results

After training, the agent reliably finds a near-optimal path from start to goal.

- Reward converges from around -700 (early random exploration) to a stable 
  ~+80 by episode 300-400
- Steps to reach the goal drop from unpredictable/timing-out to a consistent 
  ~18-19 steps
- The learned value heatmap shows correctly decreasing values near obstacles 
  and increasing values toward the goal

See the `results/` folder for generated plots:
- `reward_curve.png` — reward convergence over training
- `steps_curve.png` — steps-to-goal over training
- `value_heatmap.png` — learned state values across the grid
- `best_path.png` — the final path taken by the trained agent

## How to Run

### Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install numpy matplotlib
```

### Train and visualize (full pipeline)

```bash
python main.py
```

### Run only training

```bash
python main.py --mode train
```

### Run only visualization (uses previously saved Q-table)

```bash
python main.py --mode visualize
```

## Tech Stack

- Python
- NumPy — Q-table and numerical operations
- Matplotlib — visualization
- Git — version control

## Author

Jayakrishnan K — MCA, Major Project (21CSA699A)