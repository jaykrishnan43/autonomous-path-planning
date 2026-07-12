import streamlit as st
import numpy as np
import json
import time
from collections import deque
import matplotlib.pyplot as plt
from environment.grid_world import GridWorld
from environment.grid_world_waypoint import GridWorldWaypoint
from agent.q_learning_agent import QLearningAgent
from agent.q_learning_agent_waypoint import QLearningAgentWaypoint

st.set_page_config(page_title="Autonomous Path Planning", layout="wide")

st.markdown("""
<style>
.st-key-grid_tab2 div[data-testid="stHorizontalBlock"],
.st-key-grid_tab3 div[data-testid="stHorizontalBlock"] {
    gap: 2px !important;
}
.st-key-grid_tab2 div[data-testid="column"],
.st-key-grid_tab3 div[data-testid="column"] {
    width: 34px !important; min-width: 34px !important; max-width: 34px !important;
    flex: none !important; padding: 0px !important;
    display: flex; justify-content: center; align-items: center;
}
.st-key-grid_tab2 div[data-testid="stCheckbox"] {
    display: flex; justify-content: center; align-items: center;
    border: 1px solid #444; border-radius: 4px; width: 30px; height: 30px;
    background-color: #1e1e1e;
}
.st-key-grid_tab2 div[data-testid="stCheckbox"] label {
    display: flex; justify-content: center; align-items: center;
    width: 100%; height: 100%; margin: 0 !important;
}
.st-key-grid_tab2 div[data-testid="stCheckbox"] label div:first-child {
    margin: 0 !important;
}
.st-key-grid_tab2 div[data-testid="stMarkdownContainer"] {
    display: flex; justify-content: center; align-items: center;
    border: 1px solid #444; border-radius: 4px; width: 30px; height: 30px;
    background-color: #1e1e1e;
}
.st-key-grid_tab2 div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important; font-size: 20px; line-height: 1; text-align: center;
}
.st-key-grid_tab3 div[data-testid="stButton"] button {
    width: 32px !important; height: 32px !important;
    min-width: 32px !important; padding: 0px !important;
    font-size: 16px !important; line-height: 1 !important;
    border: 1px solid #444 !important; border-radius: 4px !important;
    background-color: #1e1e1e !important;
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = 'results/webapp_data'

SCENARIOS = {
    "Baseline (10x10)": "baseline_10x10",
    "Larger Grid (15x15)": "grid_15x15",
    "Larger Grid (20x20)": "grid_20x20",
    "Low Obstacle Density (5%)": "density_low",
    "High Obstacle Density (30%)": "density_high",
}

CUSTOM_SIZE = 10
CUSTOM_START = (0, 0)
CUSTOM_GOAL = (9, 9)
CUSTOM_EPISODES = 800
CUSTOM_MAX_STEPS = 200

WP_SIZE = 10
WP_START = (0, 0)
WP_EPISODES = 1000
WP_MAX_STEPS = 200


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

        if len(visited_recently) >= 2 and next_state == visited_recently[-2]:
            q_values = q_table[x, y].copy()
            q_values[action] = -np.inf
            action = int(np.argmax(q_values))
            next_state, reward, done, _ = env.step(action)

        path.append(next_state)
        visited_recently.append(state)
        state = next_state
        if done:
            break

    return path, done


def draw_grid(size, obstacles, start, goal, path_so_far, full_path_length, checkpoint=None):
    grid_display = np.zeros((size, size))
    for ox, oy in obstacles:
        grid_display[ox, oy] = -1

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(grid_display, cmap='gray_r')

    if len(path_so_far) > 1:
        xs = [p[1] for p in path_so_far]
        ys = [p[0] for p in path_so_far]
        ax.plot(xs, ys, color='blue', linewidth=2, alpha=0.7)

    current = path_so_far[-1]
    ax.plot(current[1], current[0], marker='o', color='blue', markersize=16, zorder=6)

    if checkpoint is not None:
        ax.scatter([checkpoint[1]], [checkpoint[0]], color='orange', s=200, label='Checkpoint', zorder=5)
    ax.scatter([start[1]], [start[0]], color='green', s=200, label='Start', zorder=5)
    ax.scatter([goal[1]], [goal[0]], color='red', s=200, label='Goal', zorder=5)
    ax.legend(loc='upper left')
    ax.set_title(f"Step {len(path_so_far) - 1} / {full_path_length - 1}")

    return fig


def is_solvable(size, start, goal, obstacles):
    obstacle_set = set(obstacles)
    visited = set([start])
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in obstacle_set and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append((nx, ny))
    return False


def train_live(obstacles):
    env = GridWorld(size=CUSTOM_SIZE, start=CUSTOM_START, goal=CUSTOM_GOAL, obstacles=obstacles)
    agent = QLearningAgent(grid_size=CUSTOM_SIZE, action_space=env.action_space)
    for episode in range(CUSTOM_EPISODES):
        state = env.reset()
        for step in range(CUSTOM_MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()
    return agent.q_table


def train_live_waypoint(obstacles, start, goal, checkpoint):
    env = GridWorldWaypoint(size=WP_SIZE, start=start, goal=goal,
                             waypoint=checkpoint, obstacles=obstacles)
    agent = QLearningAgentWaypoint(grid_size=WP_SIZE, action_space=env.action_space)
    for episode in range(WP_EPISODES):
        state = env.reset()
        for step in range(WP_MAX_STEPS):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()
    return agent.q_table


def get_path_waypoint(q_table, start, goal, checkpoint, obstacles):
    env = GridWorldWaypoint(size=WP_SIZE, start=start, goal=goal,
                             waypoint=checkpoint, obstacles=obstacles)
    state = env.reset()
    path = [(state[0], state[1])]
    visited_recently = []

    for _ in range(WP_MAX_STEPS):
        x, y, visited = state
        action = int(np.argmax(q_table[x, y, visited]))
        next_state, reward, done, _ = env.step(action)

        if len(visited_recently) >= 2 and (next_state[0], next_state[1]) == visited_recently[-2]:
            q_values = q_table[x, y, visited].copy()
            q_values[action] = -np.inf
            action = int(np.argmax(q_values))
            next_state, reward, done, _ = env.step(action)

        path.append((next_state[0], next_state[1]))
        visited_recently.append((x, y))
        state = next_state
        if done:
            break

    return path, done


# --- UI ---

st.title("Autonomous Path Planning — Q-Learning Agent")
st.markdown("""
This demo shows a **Q-learning agent** navigating a grid-world environment, 
having learned an optimal path to the goal through trial-and-error using the 
Bellman equation and epsilon-greedy exploration.
""")

tab1, tab2, tab3 = st.tabs([
    "Pre-trained Scenarios", "Custom Obstacles (Live Training)", "Waypoint & Movable Goal"
])

# --- TAB 1: Pre-trained scenarios ---
with tab1:
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
        play = st.button("▶ Play Animation", key="play_scenario")
        speed = st.slider("Speed (seconds per step)", 0.05, 0.5, 0.15, key="speed_scenario")

    with col2:
        plot_area = st.empty()
        if play:
            for i in range(len(path)):
                fig = draw_grid(config['size'], [tuple(o) for o in config['obstacles']],
                                 tuple(config['start']), tuple(config['goal']), path[:i + 1], len(path))
                plot_area.pyplot(fig, use_container_width=False)
                plt.close(fig)
                time.sleep(speed)
        else:
            fig = draw_grid(config['size'], [tuple(o) for o in config['obstacles']],
                             tuple(config['start']), tuple(config['goal']), path, len(path))
            plot_area.pyplot(fig, use_container_width=False)
            plt.close(fig)

# --- TAB 2: Custom obstacles, live training, fixed goal ---
with tab2:
    st.markdown(f"Design your own obstacle layout on a {CUSTOM_SIZE}x{CUSTOM_SIZE} grid. "
                f"🟢 Start is top-left, 🔴 Goal is bottom-right. "
                f"Check a box to place an obstacle there, then train the agent live.")

    if 'custom_obstacles' not in st.session_state:
        st.session_state.custom_obstacles = set()

    col_grid, col_controls = st.columns([1.2, 1])
    with col_grid:
        with st.container(key="grid_tab2"):
            for r in range(CUSTOM_SIZE):
                cols = st.columns(CUSTOM_SIZE)
                for c in range(CUSTOM_SIZE):
                    cell = (r, c)
                    with cols[c]:
                        if cell == CUSTOM_START:
                            st.markdown("🟢")
                        elif cell == CUSTOM_GOAL:
                            st.markdown("🔴")
                        else:
                            checked = cell in st.session_state.custom_obstacles
                            new_val = st.checkbox("", value=checked, key=f"cell_{r}_{c}", label_visibility="collapsed")
                            if new_val:
                                st.session_state.custom_obstacles.add(cell)
                            else:
                                st.session_state.custom_obstacles.discard(cell)

    with col_controls:
        st.markdown(f"**Obstacles placed:** {len(st.session_state.custom_obstacles)}")
        if st.button("Clear All Obstacles", key="clear_tab2"):
            st.session_state.custom_obstacles = set()
            st.rerun()

        if st.button("🚀 Check & Train Agent", type="primary", key="train_tab2"):
            obstacles = list(st.session_state.custom_obstacles)
            if not is_solvable(CUSTOM_SIZE, CUSTOM_START, CUSTOM_GOAL, obstacles):
                st.error("❌ No path exists from Start to Goal with this obstacle layout. "
                          "Remove some obstacles and try again.")
            else:
                with st.spinner(f"Training agent live ({CUSTOM_EPISODES} episodes)..."):
                    q_table = train_live(obstacles)
                config = {'size': CUSTOM_SIZE, 'obstacles': obstacles,
                          'start': list(CUSTOM_START), 'goal': list(CUSTOM_GOAL)}
                path, reached_goal = get_path(q_table, config)
                st.session_state.custom_result = {'path': path, 'reached_goal': reached_goal, 'obstacles': obstacles}
                st.success(f"✅ Training complete! Path length: {len(path) - 1} steps")

    if 'custom_result' in st.session_state:
        result = st.session_state.custom_result
        st.divider()
        st.subheader("Result")
        result_col, _ = st.columns([1, 1])
        with result_col:
            play_custom = st.button("▶ Play Animation", key="play_custom")
            plot_area_custom = st.empty()
            if play_custom:
                for i in range(len(result['path'])):
                    fig = draw_grid(CUSTOM_SIZE, result['obstacles'], CUSTOM_START, CUSTOM_GOAL,
                                     result['path'][:i + 1], len(result['path']))
                    plot_area_custom.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                    time.sleep(0.15)
            else:
                fig = draw_grid(CUSTOM_SIZE, result['obstacles'], CUSTOM_START, CUSTOM_GOAL,
                                 result['path'], len(result['path']))
                plot_area_custom.pyplot(fig, use_container_width=False)
                plt.close(fig)

# --- TAB 3: Waypoint checkpoint + movable start/goal, click-to-place, fixed 10x10 ---
with tab3:
    st.markdown(f"Same {WP_SIZE}x{WP_SIZE} grid. Pick a placement mode below, then click cells "
                f"to place obstacles, or move the start, goal, and checkpoint.")

    if 'wp_obstacles' not in st.session_state:
        st.session_state.wp_obstacles = set()
    if 'wp_start' not in st.session_state:
        st.session_state.wp_start = WP_START
    if 'wp_goal' not in st.session_state:
        st.session_state.wp_goal = (9, 9)
    if 'wp_checkpoint' not in st.session_state:
        st.session_state.wp_checkpoint = None

    wp_mode = st.radio("Click mode — what does clicking a cell place?",
                        ["Obstacle", "Start", "Goal", "Checkpoint"], horizontal=True, key="wp_mode_radio")

    col_grid, col_controls = st.columns([1.2, 1])

    with col_grid:
        with st.container(key="grid_tab3"):
            for r in range(WP_SIZE):
                cols = st.columns(WP_SIZE)
                for c in range(WP_SIZE):
                    cell = (r, c)
                    with cols[c]:
                        if cell == st.session_state.wp_start:
                            label = "🟢"
                        elif cell == st.session_state.wp_goal:
                            label = "🔴"
                        elif st.session_state.wp_checkpoint is not None and cell == st.session_state.wp_checkpoint:
                            label = "🟠"
                        elif cell in st.session_state.wp_obstacles:
                            label = "⬛"
                        else:
                            label = "·"

                        clicked = st.button(label, key=f"wpbtn_{r}_{c}")
                        if clicked:
                            if wp_mode == "Obstacle":
                                if cell in (st.session_state.wp_start, st.session_state.wp_goal,
                                            st.session_state.wp_checkpoint):
                                    st.toast("Can't place an obstacle on the start, goal, or checkpoint.")
                                elif cell in st.session_state.wp_obstacles:
                                    st.session_state.wp_obstacles.discard(cell)
                                else:
                                    st.session_state.wp_obstacles.add(cell)
                            elif wp_mode == "Start":
                                if cell == st.session_state.wp_goal or cell == st.session_state.wp_checkpoint:
                                    st.toast("Can't place Start on the goal or checkpoint.")
                                else:
                                    st.session_state.wp_obstacles.discard(cell)
                                    st.session_state.wp_start = cell
                            elif wp_mode == "Goal":
                                if cell == st.session_state.wp_start:
                                    st.toast("Can't place Goal on the start.")
                                else:
                                    st.session_state.wp_obstacles.discard(cell)
                                    if st.session_state.wp_checkpoint == cell:
                                        st.session_state.wp_checkpoint = None
                                    st.session_state.wp_goal = cell
                            elif wp_mode == "Checkpoint":
                                if cell == st.session_state.wp_start or cell == st.session_state.wp_goal:
                                    st.toast("Can't place a checkpoint on the start or goal.")
                                else:
                                    st.session_state.wp_obstacles.discard(cell)
                                    if st.session_state.wp_checkpoint == cell:
                                        st.session_state.wp_checkpoint = None
                                    else:
                                        st.session_state.wp_checkpoint = cell
                            st.rerun()

    with col_controls:
        st.markdown(f"**Obstacles placed:** {len(st.session_state.wp_obstacles)}")
        st.markdown(f"**Start:** {st.session_state.wp_start}")
        st.markdown(f"**Goal:** {st.session_state.wp_goal}")
        st.markdown(f"**Checkpoint:** {st.session_state.wp_checkpoint if st.session_state.wp_checkpoint else 'None set'}")

        if st.button("Clear All Obstacles", key="clear_tab3"):
            st.session_state.wp_obstacles = set()
            st.rerun()
        if st.session_state.wp_checkpoint is not None:
            if st.button("Remove Checkpoint", key="remove_cp_tab3"):
                st.session_state.wp_checkpoint = None
                st.rerun()

        st.divider()

        if st.button("🚀 Check & Train Agent", type="primary", key="train_tab3"):
            obstacles = list(st.session_state.wp_obstacles)
            start = st.session_state.wp_start
            goal = st.session_state.wp_goal
            checkpoint = st.session_state.wp_checkpoint

            solvable = (is_solvable(WP_SIZE, start, checkpoint, obstacles) and
                        is_solvable(WP_SIZE, checkpoint, goal, obstacles)) if checkpoint else \
                       is_solvable(WP_SIZE, start, goal, obstacles)

            if not solvable:
                st.error("❌ No valid path exists for this layout. Adjust obstacles, start, goal, or checkpoint.")
            else:
                with st.spinner(f"Training agent live ({WP_EPISODES} episodes)..."):
                    q_table = train_live_waypoint(obstacles, start, goal, checkpoint)
                path, reached_goal = get_path_waypoint(q_table, start, goal, checkpoint, obstacles)
                st.session_state.wp_result = {
                    'path': path, 'reached_goal': reached_goal, 'obstacles': obstacles,
                    'start': start, 'goal': goal, 'checkpoint': checkpoint
                }
                st.success(f"✅ Training complete! Path length: {len(path) - 1} steps")

    if 'wp_result' in st.session_state:
        result = st.session_state.wp_result
        st.divider()
        st.subheader("Result")
        result_col, _ = st.columns([1, 1])
        with result_col:
            play_wp = st.button("▶ Play Animation", key="play_wp")
            plot_area_wp = st.empty()
            if play_wp:
                for i in range(len(result['path'])):
                    fig = draw_grid(WP_SIZE, result['obstacles'], result['start'], result['goal'],
                                     result['path'][:i + 1], len(result['path']), checkpoint=result['checkpoint'])
                    plot_area_wp.pyplot(fig, use_container_width=False)
                    plt.close(fig)
                    time.sleep(0.15)
            else:
                fig = draw_grid(WP_SIZE, result['obstacles'], result['start'], result['goal'],
                                 result['path'], len(result['path']), checkpoint=result['checkpoint'])
                plot_area_wp.pyplot(fig, use_container_width=False)
                plt.close(fig)