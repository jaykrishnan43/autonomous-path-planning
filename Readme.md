# Autonomous Path Planning using Q-Learning

A reinforcement learning agent that learns to navigate a grid-world environment 
from a start point to a goal, avoiding obstacles, using the Q-learning algorithm 
— trained entirely through trial and error, with no prior map.

Developed as a major project for course 21CSA699A (MCA), under Independent 
Exploration mode.

**Live demo:** [INSERT YOUR STREAMLIT URL HERE]

## Overview

This project implements a tabular Q-learning agent that learns an optimal 
navigation policy through trial-and-error interaction with a grid-world 
environment. Beyond the core agent, the project includes a substantial set of 
comparative experiments, a formal evaluation of success rate and path 
efficiency, an adaptive replanning mechanism for dynamic obstacles, and a 
live, publicly deployed interactive web application.

**Core concepts demonstrated:**
- Markov Decision Processes (MDPs)
- The Bellman equation for value updates
- Epsilon-greedy exploration vs. exploitation
- Q-table convergence
- Potential-based reward shaping
- Reactive replanning under non-stationary conditions

## Project Structure
autonomous-path-planning/
├── environment/
│   ├── grid_world.py            Static-obstacle grid environment
│   └── dynamic_grid_world.py    Environment with moving obstacles
├── agent/
│   └── q_learning_agent.py      Q-table, epsilon-greedy policy, Bellman update
├── training/
│   └── train.py                 Standalone training script
├── visualization/
│   ├── visualize.py             Static result plots
│   └── animate.py               Real-time training/path animations
├── experiments/
│   ├── grid_size_comparison.py
│   ├── obstacle_density_comparison.py
│   ├── obstacle_density_20x20.py
│   ├── epsilon_comparison.py
│   ├── reward_shaping_comparison.py
│   ├── success_rate_and_efficiency.py
│   ├── dynamic_obstacles_demo.py
│   └── prepare_webapp_data.py
├── results/                     Saved Q-tables, metrics, and plots
├── app.py                       Streamlit web application
├── main.py                      Main entry point
└── README.md
## How It Works

1. **Environment**: A 10x10 grid with a defined start, goal, and obstacles.
   The agent receives +100 for reaching the goal, -50 for hitting an obstacle,
   -5 for hitting a boundary, and -1 for every other step (encouraging shorter paths).

2. **Agent**: Uses a Q-table to estimate the value of each (state, action) pair.
   Actions are chosen using an epsilon-greedy policy, with epsilon decaying
   over the course of training.

3. **Learning**: The Q-table is updated after every step using the Bellman equation:

Q(s,a) ← Q(s,a) + α [r + γ · max(Q(s',a')) − Q(s,a)]

4. **Training**: The agent runs for 1000 episodes, gradually shifting from 
   random exploration to exploiting its learned policy.

## Key Experiments & Findings

- **Grid size** (10×10 / 15×15 / 20×20): convergence time scales with state 
  space size — the 20×20 grid had not fully converged within 1000 episodes.
- **Obstacle density**: final policy quality stays stable across densities, 
  but an unplanned finding emerged — solvability of random layouts collapses 
  sharply near a critical density, consistent with 2D site percolation theory 
  (attempts needed to find a solvable layout: 1 → 3 → 51 → failed at 55%).
- **Epsilon decay schedules**: decay rate must be calibrated to the training 
  budget; too slow a decay leaves the agent still exploring randomly late 
  into training.
- **Reward shaping**: potential-based shaping reaches target performance 
  ~12% faster than sparse rewards, with no change to the final policy.
- **Success rate & path efficiency**: the trained baseline agent achieves a 
  100% success rate and 100% path efficiency (18 steps, matching the 
  theoretical shortest possible path).
- **Dynamic obstacles**: a reactive replanning layer reduces obstacle 
  collisions from 2.48 to 0.00 per trial compared to a naive static policy.

## Live Web Application

A Streamlit app is deployed publicly, offering:
- **Pre-trained Scenarios** — replay 5 trained configurations with animated playback
- **Custom Obstacles** — design your own layout by clicking cells; trains a fresh agent live in seconds
- **Waypoint & Movable Goal** — reposition the start, goal, and an optional checkpoint the agent must visit first

## How to Run

### Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Train and visualize (full pipeline)

```bash
python main.py
```

### Run only training / only visualization

```bash
python main.py --mode train
python main.py --mode visualize
```

### Run an experiment

```bash
python experiments/grid_size_comparison.py
```

### Run the web application locally

```bash
streamlit run app.py
```

## Tech Stack

- Python
- NumPy — Q-table and numerical operations
- Matplotlib — visualization and animation
- Streamlit — live interactive web application
- Git / GitHub — version control

## Author

Jayakrishnan K — MCA, Major Project (21CSA699A)