import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from environment.grid_world_waypoint import GridWorldWaypoint
from agent.q_learning_agent_waypoint import QLearningAgentWaypoint

GRID_SIZE = 10
OBSTACLES = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
START = (0, 0)
GOAL = (9, 9)
WAYPOINT = (7, 2)  # must be touched before the goal counts
NUM_EPISODES = 1500
MAX_STEPS_PER_EPISODE = 300


def train():
    env = GridWorldWaypoint(size=GRID_SIZE, start=START, goal=GOAL,
                             waypoint=WAYPOINT, obstacles=OBSTACLES)
    agent = QLearningAgentWaypoint(grid_size=GRID_SIZE, action_space=env.action_space)

    episode_rewards, episode_steps = [], []

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

        if (episode + 1) % 150 == 0:
            print(f"Episode {episode + 1}/{NUM_EPISODES} | "
                  f"Avg Reward (last 150): {np.mean(episode_rewards[-150:]):.2f} | "
                  f"Avg Steps (last 150): {np.mean(episode_steps[-150:]):.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    print("\nTraining complete!")
    return agent, episode_rewards, episode_steps


def get_path(agent):
    env = GridWorldWaypoint(size=GRID_SIZE, start=START, goal=GOAL,
                             waypoint=WAYPOINT, obstacles=OBSTACLES)
    state = env.reset()
    path = [state]
    for _ in range(MAX_STEPS_PER_EPISODE):
        x, y, visited = state
        action = int(np.argmax(agent.q_table[x, y, visited]))
        state, reward, done, _ = env.step(action)
        path.append(state)
        if done:
            break
    return path, done


def plot_results(episode_rewards, episode_steps, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    window = 20

    axes[0].plot(np.convolve(episode_rewards, np.ones(window)/window, mode='valid'))
    axes[0].set_title('Reward Convergence (Waypoint Task)')
    axes[0].set_xlabel('Episode'); axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.convolve(episode_steps, np.ones(window)/window, mode='valid'))
    axes[1].set_title('Steps Convergence (Waypoint Task)')
    axes[1].set_xlabel('Episode'); axes[1].grid(True, alpha=0.3)

    grid_display = np.zeros((GRID_SIZE, GRID_SIZE))
    for ox, oy in OBSTACLES:
        grid_display[ox, oy] = -1
    axes[2].imshow(grid_display, cmap='gray_r')
    xs = [p[1] for p in path]; ys = [p[0] for p in path]
    axes[2].plot(xs, ys, marker='o', color='blue', linewidth=2, markersize=5)
    axes[2].scatter([START[1]], [START[0]], color='green', s=200, label='Start', zorder=5)
    axes[2].scatter([WAYPOINT[1]], [WAYPOINT[0]], color='orange', s=200, label='Checkpoint', zorder=5)
    axes[2].scatter([GOAL[1]], [GOAL[0]], color='red', s=200, label='Goal', zorder=5)
    axes[2].legend(loc='upper left')
    axes[2].set_title('Learned Path: Start \u2192 Checkpoint \u2192 Goal')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/waypoint_navigation.png')
    print("Saved results/waypoint_navigation.png")
    plt.close()


if __name__ == "__main__":
    agent, episode_rewards, episode_steps = train()
    path, reached_goal = get_path(agent)
    checkpoint_visited = any((p[0], p[1]) == WAYPOINT for p in path)
    print(f"\nFinal path length: {len(path) - 1} steps")
    print(f"Reached goal: {reached_goal}")
    print(f"Checkpoint was visited: {checkpoint_visited}")
    plot_results(episode_rewards, episode_steps, path)