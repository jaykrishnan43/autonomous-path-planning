import numpy as np
import random

class DynamicGridWorld:
    def __init__(self, size=10, start=(0, 0), goal=(9, 9), obstacles=None,
                 obstacle_move_prob=0.3, seed=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.initial_obstacles = list(obstacles) if obstacles else []
        self.obstacles = list(self.initial_obstacles)
        self.obstacle_move_prob = obstacle_move_prob
        self.state = start
        self.rng = random.Random(seed)

        self.action_space = 4
        self.action_effects = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self):
        self.state = self.start
        self.obstacles = list(self.initial_obstacles)
        return self.state

    def get_obstacles(self):
        return list(self.obstacles)

    def _move_obstacles(self):
        """Each obstacle has a chance to randomly shift to an adjacent free cell."""
        new_obstacles = []
        occupied = set(self.obstacles)

        for (ox, oy) in self.obstacles:
            if self.rng.random() < self.obstacle_move_prob:
                candidates = []
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = ox + dx, oy + dy
                    if (0 <= nx < self.size and 0 <= ny < self.size
                            and (nx, ny) != self.start and (nx, ny) != self.goal
                            and (nx, ny) not in occupied and (nx, ny) != self.state):
                        candidates.append((nx, ny))
                if candidates:
                    new_pos = self.rng.choice(candidates)
                    occupied.discard((ox, oy))
                    occupied.add(new_pos)
                    new_obstacles.append(new_pos)
                    continue
            new_obstacles.append((ox, oy))

        self.obstacles = new_obstacles

    def step(self, action):
        dx, dy = self.action_effects[action]
        x, y = self.state
        new_x, new_y = x + dx, y + dy

        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            new_x, new_y = x, y
            reward = -5
            done = False
        elif (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y
            reward = -50
            done = False
        elif (new_x, new_y) == self.goal:
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        self.state = (new_x, new_y)

        if not done:
            self._move_obstacles()  # obstacles shift after the agent moves

        return self.state, reward, done, {}

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)
        for ox, oy in self.obstacles:
            grid[ox][oy] = '#'
        gx, gy = self.goal
        grid[gx][gy] = 'G'
        sx, sy = self.state
        grid[sx][sy] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print()