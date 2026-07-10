import numpy as np

class GridWorld:
    def __init__(self, size=10, start=(0, 0), goal=(9, 9), obstacles=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = obstacles if obstacles else []
        self.state = start

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.action_space = 4
        self.action_effects = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self):
        self.state = self.start
        return self.state

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