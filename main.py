import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse
import numpy as np
from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent
from visualization import visualize

# Project configuration
GRID_SIZE = 10
OBSTACLES = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
START = (0, 0)
GOAL = (9, 9)
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200


def train():
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
    agent = QLearningAgent(grid_size=GRID_SIZE, action_space=env.action_space)

    episode_rewards = []
    episode_steps = []
    q_snapshots = {}  # stores Q-table copies at intervals, for animating training progress

    print("Starting training...\n")

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

        # Save a snapshot of the Q-table every 50 episodes
        if (episode + 1) % 50 == 0:
            q_snapshots[episode + 1] = agent.q_table.copy()

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_steps = np.mean(episode_steps[-100:])
            print(f"Episode {episode + 1}/{NUM_EPISODES} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Avg Steps (last 100): {avg_steps:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\nTraining complete!")

    os.makedirs('results', exist_ok=True)
    np.save('results/q_table.npy', agent.q_table)
    np.save('results/episode_rewards.npy', np.array(episode_rewards))
    np.save('results/episode_steps.npy', np.array(episode_steps))

    # Save all snapshots into one compressed file
    np.savez('results/q_snapshots.npz', **{str(k): v for k, v in q_snapshots.items()})

    print("Q-table, metrics, and training snapshots saved to results/ folder.")


def visualize_results():
    print("\nGenerating visualizations...\n")
    visualize.plot_rewards()
    visualize.plot_steps()
    visualize.plot_value_heatmap()
    visualize.plot_best_path()
    print("\nAll visualizations complete. Check the results/ folder.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous Path Planning - Q-Learning Agent")
    parser.add_argument('--mode', type=str, default='all',
                         choices=['train', 'visualize', 'all'],
                         help="Choose 'train', 'visualize', or 'all' (default: all)")
    args = parser.parse_args()

    if args.mode in ('train', 'all'):
        train()

    if args.mode in ('visualize', 'all'):
        visualize_results()