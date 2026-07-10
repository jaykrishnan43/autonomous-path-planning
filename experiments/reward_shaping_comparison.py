import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent

GRID_SIZE = 10
OBSTACLES = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
START = (0, 0)
GOAL = (9, 9)
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200
GAMMA = 0.9


def manhattan_distance(state, goal):
    return abs(state[0] - goal[0]) + abs(state[1] - goal[1])


def potential(state, goal):
    # Higher potential = closer to goal
    return -manhattan_distance(state, goal)


def train(use_shaping):
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
    agent = QLearningAgent(grid_size=GRID_SIZE, action_space=env.action_space, gamma=GAMMA)

    episode_rewards = []  # tracks the ORIGINAL environment reward, for fair comparison
    episode_steps = []

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)

            if use_shaping:
                shaping_bonus = GAMMA * potential(next_state, GOAL) - potential(state, GOAL)
                learning_reward = reward + shaping_bonus
            else:
                learning_reward = reward

            agent.update(state, action, learning_reward, next_state, done)

            state = next_state
            total_reward += reward  # always track the true environment reward
            if done:
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)

    return episode_rewards, episode_steps


def run_comparison():
    print("Training WITHOUT reward shaping (sparse)...")
    sparse_rewards, sparse_steps = train(use_shaping=False)
    print(f"  Done. Final avg reward: {np.mean(sparse_rewards[-100:]):.2f}, "
          f"final avg steps: {np.mean(sparse_steps[-100:]):.2f}\n")

    print("Training WITH reward shaping (potential-based)...")
    shaped_rewards, shaped_steps = train(use_shaping=True)
    print(f"  Done. Final avg reward: {np.mean(shaped_rewards[-100:]):.2f}, "
          f"final avg steps: {np.mean(shaped_steps[-100:]):.2f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 20

    for rewards, label in [(sparse_rewards, 'Sparse (original)'), (shaped_rewards, 'Shaped (potential-based)')]:
        rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
        axes[0].plot(rolling, label=label)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('True Environment Reward (20-episode rolling avg)')
    axes[0].set_title('Reward Convergence: Sparse vs. Shaped')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for steps, label in [(sparse_steps, 'Sparse (original)'), (shaped_steps, 'Shaped (potential-based)')]:
        rolling = np.convolve(steps, np.ones(window) / window, mode='valid')
        axes[1].plot(rolling, label=label)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal (20-episode rolling avg)')
    axes[1].set_title('Steps Convergence: Sparse vs. Shaped')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/reward_shaping_comparison.png')
    print("Saved results/reward_shaping_comparison.png")
    plt.close()

    # Find first episode where each version reaches a "good" performance threshold
    threshold = 50
    def first_episode_above(rewards, threshold):
        rolling = np.convolve(rewards, np.ones(window) / window, mode='valid')
        above = np.where(rolling >= threshold)[0]
        return int(above[0]) if len(above) > 0 else None

    sparse_first = first_episode_above(sparse_rewards, threshold)
    shaped_first = first_episode_above(shaped_rewards, threshold)

    print("\n--- Summary ---")
    print(f"Episodes to first reach reward >= {threshold}:")
    print(f"  Sparse: {sparse_first}")
    print(f"  Shaped: {shaped_first}")


if __name__ == "__main__":
    run_comparison()