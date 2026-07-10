import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from environment.grid_world import GridWorld

GRID_SIZE = 10
OBSTACLES = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
START = (0, 0)
GOAL = (9, 9)


def get_path_from_qtable(q_table, max_steps=100):
    """Runs the agent greedily using a given Q-table and returns the path taken."""
    env = GridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES)
    state = env.reset()
    path = [state]

    for _ in range(max_steps):
        x, y = state
        action = int(np.argmax(q_table[x, y]))
        state, reward, done, _ = env.step(action)
        path.append(state)
        if done:
            break

    return path


def draw_grid_base(ax):
    """Draws the static grid background: obstacles, start, goal."""
    grid_display = np.zeros((GRID_SIZE, GRID_SIZE))
    for ox, oy in OBSTACLES:
        grid_display[ox, oy] = -1

    ax.imshow(grid_display, cmap='gray_r')
    ax.scatter([START[1]], [START[0]], color='green', s=200, label='Start', zorder=5)
    ax.scatter([GOAL[1]], [GOAL[0]], color='red', s=200, label='Goal', zorder=5)
    ax.legend(loc='upper left')


def animate_final_path():
    """Animates the trained agent moving step-by-step along its final learned path."""
    q_table = np.load('results/q_table.npy')
    path = get_path_from_qtable(q_table)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_grid_base(ax)
    ax.set_title('Agent Navigation (Final Trained Policy)')

    agent_dot, = ax.plot([], [], marker='o', color='blue', markersize=14, zorder=6)
    trail_line, = ax.plot([], [], color='blue', linewidth=2, alpha=0.6)

    def update(frame):
        current_path = path[:frame + 1]
        xs = [p[1] for p in current_path]
        ys = [p[0] for p in current_path]
        trail_line.set_data(xs, ys)
        agent_dot.set_data([xs[-1]], [ys[-1]])
        return agent_dot, trail_line

    anim = animation.FuncAnimation(fig, update, frames=len(path), interval=300, blit=True, repeat=True)
    anim.save('results/final_path_animation.gif', writer='pillow', fps=3)
    print("Saved results/final_path_animation.gif")
    plt.close()


def animate_training_progress():
    """Animates how the agent's best path evolves across training snapshots."""
    snapshots = np.load('results/q_snapshots.npz')
    episode_keys = sorted(snapshots.files, key=lambda k: int(k))

    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        ax.clear()
        draw_grid_base(ax)
        episode_key = episode_keys[frame_idx]
        q_table = snapshots[episode_key]
        path = get_path_from_qtable(q_table)

        xs = [p[1] for p in path]
        ys = [p[0] for p in path]
        ax.plot(xs, ys, marker='o', color='blue', linewidth=2, markersize=6)
        ax.set_title(f'Learned Path at Episode {episode_key} (Path length: {len(path)})')

    anim = animation.FuncAnimation(fig, update, frames=len(episode_keys), interval=500, repeat=True)
    anim.save('results/training_progress_animation.gif', writer='pillow', fps=2)
    print("Saved results/training_progress_animation.gif")
    plt.close()


if __name__ == "__main__":
    print("Generating animations... this may take a minute.\n")
    animate_final_path()
    animate_training_progress()
    print("\nAnimations complete. Check the results/ folder.")