import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 300


def scale_obstacles(base_obstacles, base_size, new_size):
    """Scales obstacle positions proportionally when grid size changes."""
    scale = new_size / base_size
    scaled = set()
    for ox, oy in base_obstacles:
        scaled.add((int(ox * scale), int(oy * scale)))
    return list(scaled)


def train_on_grid(grid_size):
    base_obstacles = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
    obstacles = scale_obstacles(base_obstacles, base_size=10, new_size=grid_size)

    start = (0, 0)
    goal = (grid_size - 1, grid_size - 1)

    env = GridWorld(size=grid_size, start=start, goal=goal, obstacles=obstacles)
    agent = QLearningAgent(grid_size=grid_size, action_space=env.action_space)

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
    grid_sizes = [10, 15, 20]
    results = {}

    for size in grid_sizes:
        print(f"Training on {size}x{size} grid...")
        rewards, steps = train_on_grid(size)
        results[size] = {'rewards': rewards, 'steps': steps}
        avg_final_reward = np.mean(rewards[-100:])
        avg_final_steps = np.mean(steps[-100:])
        print(f"  {size}x{size} done. Final avg reward: {avg_final_reward:.2f}, "
              f"final avg steps: {avg_final_steps:.2f}\n")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for size in grid_sizes:
        rewards = results[size]['rewards']
        window = 20
        rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(rolling, label=f'{size}x{size}')

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (20-episode rolling avg)')
    axes[0].set_title('Reward Convergence by Grid Size')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for size in grid_sizes:
        steps = results[size]['steps']
        window = 20
        rolling = np.convolve(steps, np.ones(window) / window, mode='valid')
        axes[1].plot(rolling, label=f'{size}x{size}')

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal (20-episode rolling avg)')
    axes[1].set_title('Steps Convergence by Grid Size')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/grid_size_comparison.png')
    print("Saved results/grid_size_comparison.png")
    plt.close()

    # Print summary table
    print("\n--- Summary ---")
    print(f"{'Grid Size':<12}{'Final Avg Reward':<20}{'Final Avg Steps':<20}")
    for size in grid_sizes:
        r = np.mean(results[size]['rewards'][-100:])
        s = np.mean(results[size]['steps'][-100:])
        print(f"{f'{size}x{size}':<12}{r:<20.2f}{s:<20.2f}")


if __name__ == "__main__":
    run_comparison()