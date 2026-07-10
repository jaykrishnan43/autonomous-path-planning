import streamlit as st
import numpy as np
import json
import time
import matplotlib.pyplot as plt
from environment.grid_world import GridWorld

st.set_page_config(page_title="Autonomous Path Planning", layout="wide")

DATA_DIR = 'results/webapp_data'

SCENARIOS = {
    "Baseline (10x10)": "baseline_10x10",
    "Larger Grid (15x15)": "grid_15x15",
    "Larger Grid (20x20)": "grid_20x20",
    "Low Obstacle Density (5%)": "density_low",
    "High Obstacle Density (30%)": "density_high",
}


@st.cache_data
def load_scenario(name):
    q_table = np.load(f'{DATA_DIR}/{name}_qtable.npy')
    with open(f'{DATA_DIR}/{name}_config.json') as f:
        config = json.load(f)
    return q_table, config


def get_path(q_table, config):
    env = GridWorld(size=config['size'], start=tuple(config['start']),
                     goal=tuple(config['goal']), obstacles=[tuple(o) for o in config['obstacles']])
    state = env.reset()
    path = [state]
    visited_recently = []

    max_steps = config['size'] * config['size'] * 2

    for _ in range(max_steps):
        x, y = state
        action = int(np.argmax(q_table[x, y]))
        next_state, reward, done, _ = env.step(action)

        # Detect a 2-state oscillation loop (A -> B -> A -> B ...)
        if len(visited_recently) >= 2 and next_state == visited_recently[-2]:
            q_values = q_table[x, y].copy()
            q_values[action] = -np.inf  # rule out the looping action
            action = int(np.argmax(q_values))
            next_state, reward, done, _ = env.step(action)

        path.append(next_state)
        visited_recently.append(state)
        state = next_state

        if done:
            break

    return path, done


def draw_grid(config, path_so_far, full_path_length):
    size = config['size']
    obstacles = [tuple(o) for o in config['obstacles']]
    start = tuple(config['start'])
    goal = tuple(config['goal'])

    grid_display = np.zeros((size, size))
    for ox, oy in obstacles:
        grid_display[ox, oy] = -1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid_display, cmap='gray_r')

    if len(path_so_far) > 1:
        xs = [p[1] for p in path_so_far]
        ys = [p[0] for p in path_so_far]
        ax.plot(xs, ys, color='blue', linewidth=2, alpha=0.7)

    current = path_so_far[-1]
    ax.plot(current[1], current[0], marker='o', color='blue', markersize=16, zorder=6)

    ax.scatter([start[1]], [start[0]], color='green', s=200, label='Start', zorder=5)
    ax.scatter([goal[1]], [goal[0]], color='red', s=200, label='Goal', zorder=5)
    ax.legend(loc='upper left')
    ax.set_title(f"Step {len(path_so_far) - 1} / {full_path_length - 1}")

    return fig


# --- UI ---

st.title("Autonomous Path Planning — Q-Learning Agent")
st.markdown("""
This demo shows a **Q-learning agent** navigating a grid-world environment, 
having learned an optimal path to the goal through trial-and-error using the 
Bellman equation and epsilon-greedy exploration.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Scenario")
    scenario_label = st.selectbox("Choose a trained scenario", list(SCENARIOS.keys()))
    scenario_key = SCENARIOS[scenario_label]
    q_table, config = load_scenario(scenario_key)

    path, reached_goal = get_path(q_table, config)

    st.markdown(f"**Grid size:** {config['size']}x{config['size']}")
    st.markdown(f"**Obstacles:** {len(config['obstacles'])}")
    st.markdown(f"**Path length:** {len(path) - 1} steps")
    st.markdown(f"**Reached goal:** {'Yes' if reached_goal else 'No'}")

    st.divider()
    st.subheader("Playback")
    play = st.button("▶ Play Animation")
    speed = st.slider("Speed (seconds per step)", 0.05, 0.5, 0.15)

with col2:
    plot_area = st.empty()

    if play:
        for i in range(len(path)):
            fig = draw_grid(config, path[:i + 1], len(path))
            plot_area.pyplot(fig)
            plt.close(fig)
            time.sleep(speed)
    else:
        fig = draw_grid(config, path, len(path))
        plot_area.pyplot(fig)
        plt.close(fig)