import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from environment.dynamic_grid_world import DynamicGridWorld

GRID_SIZE = 10
OBSTACLES = [(2, 2), (2, 3), (3, 2), (5, 5), (6, 5), (7, 5)]
START = (0, 0)
GOAL = (9, 9)
MAX_STEPS = 100
NUM_TRIALS = 200
OBSTACLE_MOVE_PROB = 0.3


def run_trial(q_table, use_replanning, seed):
    env = DynamicGridWorld(size=GRID_SIZE, start=START, goal=GOAL, obstacles=OBSTACLES,
                            obstacle_move_prob=OBSTACLE_MOVE_PROB, seed=seed)
    state = env.reset()
    collisions = 0

    for step in range(MAX_STEPS):
        x, y = state
        q_values = q_table[x, y]
        ranked_actions = np.argsort(q_values)[::-1]  # best action first

        if use_replanning:
            # Check current (moved) obstacle positions before committing to an action
            current_obstacles = set(env.get_obstacles())
            chosen_action = None
            for a in ranked_actions:
                dx, dy = env.action_effects[a]
                nx, ny = x + dx, y + dy
                if (nx, ny) not in current_obstacles:
                    chosen_action = a
                    break
            if chosen_action is None:
                chosen_action = ranked_actions[0]  # all options blocked, take the best anyway
        else:
            # Naive: always blindly follow the originally-learned best action
            chosen_action = ranked_actions[0]

        next_state, reward, done, _ = env.step(chosen_action)
        if reward == -50:
            collisions += 1

        state = next_state
        if done:
            return True, step + 1, collisions

    return False, MAX_STEPS, collisions


def run_comparison():
    q_table = np.load('results/q_table.npy')

    print(f"Testing trained agent in a DYNAMIC environment "
          f"(obstacles move with {int(OBSTACLE_MOVE_PROB*100)}% probability per step)\n")

    for label, use_replanning in [('Naive (no replanning)', False), ('Reactive replanning', True)]:
        successes = 0
        steps_list = []
        collisions_list = []

        for trial in range(NUM_TRIALS):
            success, steps, collisions = run_trial(q_table, use_replanning, seed=trial)
            if success:
                successes += 1
                steps_list.append(steps)
            collisions_list.append(collisions)

        success_rate = successes / NUM_TRIALS * 100
        avg_steps = np.mean(steps_list) if steps_list else None
        avg_collisions = np.mean(collisions_list)

        print(f"--- {label} ---")
        print(f"Success rate: {success_rate:.1f}%")
        if avg_steps:
            print(f"Average steps (successful trials): {avg_steps:.2f}")
        else:
            print("No successful trials")
        print(f"Average obstacle collisions per trial: {avg_collisions:.2f}\n")


if __name__ == "__main__":
    run_comparison()