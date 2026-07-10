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


def train_with_epsilon_decay(epsilon_decay):
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
    agent = QLearningAgent(grid_size=GRID_SIZE, action_space=env.action_space,
                            epsilon_decay=epsilon_decay)

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
    # Different decay rates = different exploration schedules
    # Slower decay (closer to 1.0) = explores longer before settling
    decay_settings = {
        'Fast decay (0.98)': 0.98,
        'Medium decay (0.995)': 0.995,
        'Slow decay (0.999)': 0.999,
    }

    results = {}

    for label, decay in decay_settings.items():
        print(f"Training with {label}...")
        rewards, steps = train_with_epsilon_decay(decay)
        results[label] = {'rewards': rewards, 'steps': steps}
        print(f"  Done. Final avg reward: {np.mean(rewards[-100:]):.2f}, "
              f"final avg steps: {np.mean(steps[-100:]):.2f}\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    window = 20

    for label in decay_settings:
        rolling = np.convolve(results[label]['rewards'], np.ones(window) / window, mode='valid')
        axes[0].plot(rolling, label=label)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Reward (20-episode rolling avg)')
    axes[0].set_title('Reward Convergence by Epsilon Decay Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for label in decay_settings:
        rolling = np.convolve(results[label]['steps'], np.ones(window) / window, mode='valid')
        axes[1].plot(rolling, label=label)

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps to Goal (20-episode rolling avg)')
    axes[1].set_title('Steps Convergence by Epsilon Decay Rate')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/epsilon_comparison.png')
    print("Saved results/epsilon_comparison.png")
    plt.close()

    print("\n--- Summary ---")
    print(f"{'Setting':<25}{'Final Avg Reward':<20}{'Final Avg Steps':<20}")
    for label in decay_settings:
        r = np.mean(results[label]['rewards'][-100:])
        s = np.mean(results[label]['steps'][-100:])
        print(f"{label:<25}{r:<20.2f}{s:<20.2f}")


if __name__ == "__main__":
    run_comparison()