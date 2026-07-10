import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import numpy as np
import matplotlib.pyplot as plt
from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent
from collections import deque

GRID_SIZE = 10
START = (0, 0)
GOAL = (9, 9)
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 300
SEED = 42


def is_solvable(size, start, goal, obstacles):
    """BFS check to confirm a path exists from start to goal."""
    obstacle_set = set(obstacles)
    visited = set([start])
    queue = deque([start])

    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in obstacle_set and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))

    return False


def generate_obstacles(size, start, goal, density, seed):
    """Randomly generates obstacles at a given density (0.0 to 1.0), guaranteeing solvability."""
    rng = random.Random(seed)
    total_cells = size * size
    num_obstacles = int(total_cells * density)

    all_cells = [(x, y) for x in range(size) for y in range(size) if (x, y) != start and (x, y) != goal]

    attempt = 0
    while True:
        attempt += 1
        obstacles = rng.sample(all_cells, min(num_obstacles, len(all_cells)))
        if is_solvable(size, start, goal, obstacles):
            return obstacles
        if attempt > 50:
            raise RuntimeError(f"Could not generate solvable layout at density {density} after 50 attempts")


def train_with_obstacles(obstacles):
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=obstacles)
    agent = QLearningAgent(grid_size=GRID_SIZE, action_space=env.action_space)

    episode_rewards = []
    episode_steps = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)

    return episode_rewards, episode_steps


def run_comparison():
    densities = [0.05, 0.15, 0.30]  # 5%, 15%, 30% of grid cells are obstacles
    results = {}

    for density in densities:
        obstacles = generate_obstacles(GRID_SIZE, START, GOAL, density, seed=SEED)
        print(f"Training with {int(density*100)}% obstacle density ({len(obstacles)} obstacles)...")
        rewards, steps = train_with_obstacles(obstacles)
        results[density] = {'rewards': rewards, 'steps': steps, 'obstacles': obstacles}
        avg_final_reward = np.mean(rewards[-100:])
        avg_final_steps = np.mean(steps[-100:])
        print(f"  Done. Final avg reward: {avg_final_reward:.2f}, final avg steps: {avg_final_steps:.2f}\n")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for density in densities:
        rewards = results[density]['rewards']
        window = 20
        rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(rolling, label=f'{int(density*100)}% density')

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (20-episode rolling avg)')
    axes[0].set_title('Reward Convergence by Obstacle Density')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for density in densities:
        steps = results[density]['steps']
        window = 20
        rolling = np.convolve(steps, np.ones(window) / window, mode='valid')
        axes[1].plot(rolling, label=f'{int(density*100)}% density')

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal (20-episode rolling avg)')
    axes[1].set_title('Steps Convergence by Obstacle Density')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/obstacle_density_comparison.png')
    print("Saved results/obstacle_density_comparison.png")
    plt.close()

    print("\n--- Summary ---")
    print(f"{'Density':<12}{'Obstacles':<12}{'Final Avg Reward':<20}{'Final Avg Steps':<20}")
    for density in densities:
        r = np.mean(results[density]['rewards'][-100:])
        s = np.mean(results[density]['steps'][-100:])
        n = len(results[density]['obstacles'])
        print(f"{f'{int(density*100)}%':<12}{n:<12}{r:<20.2f}{s:<20.2f}")


if __name__ == "__main__":
    run_comparison()