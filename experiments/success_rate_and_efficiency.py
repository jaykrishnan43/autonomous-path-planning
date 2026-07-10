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

# Manhattan distance = theoretical shortest possible path length in an obstacle-free grid
THEORETICAL_SHORTEST = abs(GOAL[0] - START[0]) + abs(GOAL[1] - START[1])


def train_and_track():
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
    agent = QLearningAgent(grid_size=GRID_SIZE, action_space=env.action_space)

    episode_rewards = []
    episode_steps = []
    episode_success = []  # True/False did it reach the goal (vs. timing out)

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        reached_goal = False

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                reached_goal = True
                break

        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        episode_steps.append(step + 1)
        episode_success.append(reached_goal)

    return agent, episode_rewards, episode_steps, episode_success


def evaluate_final_policy(agent, num_eval_episodes=100):
    """Runs the trained agent greedily (no exploration) to measure true final performance."""
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
    successes = 0
    path_lengths = []

    for _ in range(num_eval_episodes):
        state = env.reset()
        for step in range(MAX_STEPS_PER_EPISODE):
            x, y = state
            action = int(np.argmax(agent.q_table[x, y]))  # pure exploitation, no randomness
            state, reward, done, _ = env.step(action)
            if done:
                successes += 1
                path_lengths.append(step + 1)
                break

    success_rate = successes / num_eval_episodes
    avg_path_length = np.mean(path_lengths) if path_lengths else None
    efficiency = THEORETICAL_SHORTEST / avg_path_length if avg_path_length else None

    return success_rate, avg_path_length, efficiency


def run():
    print("Training agent...\n")
    agent, episode_rewards, episode_steps, episode_success = train_and_track()

    # Success rate during training, in rolling windows of 100 episodes
    window = 100
    rolling_success_rate = [
        np.mean(episode_success[max(0, i - window):i + 1]) * 100
        for i in range(len(episode_success))
    ]

    plt.figure(figsize=(10, 5))
    plt.plot(rolling_success_rate)
    plt.xlabel('Episode')
    plt.ylabel('Success Rate % (last 100 episodes)')
    plt.title('Goal-Reaching Success Rate Over Training')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 105)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/success_rate_curve.png')
    print("Saved results/success_rate_curve.png")
    plt.close()

    print("Evaluating final trained policy (greedy, no exploration, 100 test episodes)...\n")
    success_rate, avg_path_length, efficiency = evaluate_final_policy(agent)

    print("--- Final Policy Evaluation ---")
    print(f"Theoretical shortest path (Manhattan distance): {THEORETICAL_SHORTEST} steps")
    print(f"Success rate: {success_rate * 100:.1f}%")
    print(f"Average path length achieved: {avg_path_length:.2f} steps")
    print(f"Path efficiency: {efficiency * 100:.1f}% "
          f"(theoretical shortest / actual path length)")


if __name__ == "__main__":
    run()