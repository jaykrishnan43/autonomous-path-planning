import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import numpy as np
from environment.grid_world import GridWorld
from agent.q_learning_agent import QLearningAgent

NUM_EPISODES = 1000
MAX_STEPS_PER_EPISODE = 300

OUTPUT_DIR = 'results/webapp_data'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def train_agent(size, obstacles, start, goal, episodes=NUM_EPISODES):
    env = GridWorld(size=size, start=start, goal=goal, obstacles=obstacles)
    agent = QLearningAgent(grid_size=size, action_space=env.action_space)

    for episode in range(episodes):
        state = env.reset()
        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
        agent.decay_epsilon()

    return agent.q_table


def save_scenario(name, size, obstacles, start, goal, q_table):
    np.save(f'{OUTPUT_DIR}/{name}_qtable.npy', q_table)
    config = {
        'name': name,
        'size': size,
        'obstacles': obstacles,
        'start': list(start),
        'goal': list(goal)
    }
    with open(f'{OUTPUT_DIR}/{name}_config.json', 'w') as f:
        json.dump(config, f)
    print(f"Saved scenario: {name}")


def run():
    # Scenario 1: baseline 10x10
    base_obstacles = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
    print("Training: baseline (10x10)...")
    q = train_agent(10, base_obstacles, (0, 0), (9, 9))
    save_scenario('baseline_10x10', 10, base_obstacles, (0, 0), (9, 9), q)

    # Scenario 2: 15x15 grid
    def scale_obstacles(base_obstacles, base_size, new_size):
        scale = new_size / base_size
        scaled = set()
        for ox, oy in base_obstacles:
            scaled.add((int(ox * scale), int(oy * scale)))
        return list(scaled)

    print("Training: 15x15 grid...")
    obstacles_15 = scale_obstacles(base_obstacles, 10, 15)
    q = train_agent(15, obstacles_15, (0, 0), (14, 14))
    save_scenario('grid_15x15', 15, obstacles_15, (0, 0), (14, 14), q)

    # Scenario 3: 20x20 grid
    print("Training: 20x20 grid (extended episodes for full convergence)...")
    obstacles_20 = scale_obstacles(base_obstacles, 10, 20)
    q = train_agent(20, obstacles_20, (0, 0), (19, 19), episodes=3000)
    save_scenario('grid_20x20', 20, obstacles_20, (0, 0), (19, 19), q)
    

    # Scenario 4: low obstacle density (5%) - using same generation logic as before
    import random
    from collections import deque

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

    def generate_obstacles(size, start, goal, density, seed):
        rng = random.Random(seed)
        total_cells = size * size
        num_obstacles = int(total_cells * density)
        all_cells = [(x, y) for x in range(size) for y in range(size) if (x, y) != start and (x, y) != goal]
        attempt = 0
        while True:
            attempt += 1
            obstacles = rng.sample(all_cells, min(num_obstacles, len(all_cells)))
            if is_solvable(size, start, goal, obstacles):
                return obstacles
            if attempt > 50:
                raise RuntimeError("Could not generate solvable layout")

    print("Training: low density (5%)...")
    low_density_obstacles = generate_obstacles(10, (0, 0), (9, 9), 0.05, seed=42)
    q = train_agent(10, low_density_obstacles, (0, 0), (9, 9))
    save_scenario('density_low', 10, low_density_obstacles, (0, 0), (9, 9), q)

    # Scenario 5: high obstacle density (30%)
    print("Training: high density (30%)...")
    high_density_obstacles = generate_obstacles(10, (0, 0), (9, 9), 0.30, seed=42)
    q = train_agent(10, high_density_obstacles, (0, 0), (9, 9))
    save_scenario('density_high', 10, high_density_obstacles, (0, 0), (9, 9), q)

    print("\nAll webapp data prepared successfully.")


if __name__ == "__main__":
    run()