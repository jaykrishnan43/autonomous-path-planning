import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent
import numpy as np

# Setup
GRID_SIZE = 10
OBSTACLES = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
START = (0, 0)
GOAL = (9, 9)
NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 200

env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
agent = QLearningAgent(grid_size=GRID_SIZE, action_space=env.action_space)

# Tracking for later visualization
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

    # Print progress every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        avg_steps = np.mean(episode_steps[-100:])
        print(f"Episode {episode + 1}/{NUM_EPISODES} | "
              f"Avg Reward (last 100): {avg_reward:.2f} | "
              f"Avg Steps (last 100): {avg_steps:.2f} | "
              f"Epsilon: {agent.epsilon:.3f}")

print("\nTraining complete!")

# Save the Q-table and metrics for later use
np.save('results/q_table.npy', agent.q_table)
np.save('results/episode_rewards.npy', np.array(episode_rewards))
np.save('results/episode_steps.npy', np.array(episode_steps))

print("Q-table and metrics saved to results/ folder.")