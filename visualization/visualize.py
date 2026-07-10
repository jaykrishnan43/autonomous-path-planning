import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.grid_world import GridWorld

# Load saved results
q_table = np.load('results/q_table.npy')
episode_rewards = np.load('results/episode_rewards.npy')
episode_steps = np.load('results/episode_steps.npy')

GRID_SIZE = 10
OBSTACLES = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
START = (0, 0)
GOAL = (9, 9)


def plot_rewards():
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, alpha=0.4, label='Reward per episode')

    # Rolling average makes the trend easier to see
    window = 20
    rolling_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
    plt.plot(rolling_avg, label=f'{window}-episode rolling average', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Reward per Episode (Convergence Curve)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/reward_curve.png')
    print("Saved results/reward_curve.png")
    plt.close()


def plot_steps():
    plt.figure(figsize=(10, 5))
    plt.plot(episode_steps, alpha=0.4, label='Steps per episode')

    window = 20
    rolling_avg = np.convolve(episode_steps, np.ones(window) / window, mode='valid')
    plt.plot(rolling_avg, label=f'{window}-episode rolling average', linewidth=2)

    plt.xlabel('Episode')
    plt.ylabel('Steps to Goal')
    plt.title('Steps per Episode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('results/steps_curve.png')
    print("Saved results/steps_curve.png")
    plt.close()


def plot_value_heatmap():
    # Value of a state = best Q-value across all actions from that state
    value_grid = np.max(q_table, axis=2)

    plt.figure(figsize=(8, 8))
    plt.imshow(value_grid, cmap='viridis')
    plt.colorbar(label='State Value')
    plt.title('Learned State Values (Heatmap)')

    # Mark obstacles, start, goal
    for ox, oy in OBSTACLES:
        plt.text(oy, ox, 'X', ha='center', va='center', color='red', fontsize=14, fontweight='bold')
    plt.text(START[1], START[0], 'S', ha='center', va='center', color='white', fontsize=14, fontweight='bold')
    plt.text(GOAL[1], GOAL[0], 'G', ha='center', va='center', color='white', fontsize=14, fontweight='bold')

    plt.savefig('results/value_heatmap.png')
    print("Saved results/value_heatmap.png")
    plt.close()


def plot_best_path():
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
    state = env.reset()
    path = [state]

    for _ in range(100):  # safety limit
        x, y = state
        action = int(np.argmax(q_table[x, y]))
        state, reward, done, _ = env.step(action)
        path.append(state)
        if done:
            break

    grid_display = np.zeros((GRID_SIZE, GRID_SIZE))
    for ox, oy in OBSTACLES:
        grid_display[ox, oy] = -1

    plt.figure(figsize=(8, 8))
    plt.imshow(grid_display, cmap='gray_r')

    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    plt.plot(path_x, path_y, marker='o', color='blue', linewidth=2, markersize=6, label='Agent path')

    plt.scatter([START[1]], [START[0]], color='green', s=200, label='Start', zorder=5)
    plt.scatter([GOAL[1]], [GOAL[0]], color='red', s=200, label='Goal', zorder=5)

    plt.title('Best Learned Path')
    plt.legend()
    plt.savefig('results/best_path.png')
    print("Saved results/best_path.png")
    plt.close()

    print(f"\nPath length: {len(path)} steps")
    print(f"Reached goal: {done}")


if __name__ == "__main__":
    plot_rewards()
    plot_steps()
    plot_value_heatmap()
    plot_best_path()
    print("\nAll visualizations complete. Check the results/ folder.")