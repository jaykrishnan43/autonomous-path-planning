import numpy as np

class GridWorldWaypoint:
    def __init__(self, size=10, start=(0, 0), goal=(9, 9), waypoint=None, obstacles=None):
        self.size = size
        self.start = start
        self.goal = goal
        self.waypoint = waypoint  # None = no checkpoint required
        self.obstacles = obstacles if obstacles else []
        self.state = (start[0], start[1], 0)  # (x, y, visited_checkpoint)

        self.action_space = 4
        self.action_effects = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1)
        }

    def reset(self):
        self.state = (self.start[0], self.start[1], 0)
        return self.state

    def step(self, action):
        dx, dy = self.action_effects[action]
        x, y, visited = self.state
        new_x, new_y = x + dx, y + dy
        new_visited = visited

        if new_x < 0 or new_x >= self.size or new_y < 0 or new_y >= self.size:
            new_x, new_y = x, y
            reward = -5
            done = False
        elif (new_x, new_y) in self.obstacles:
            new_x, new_y = x, y
            reward = -50
            done = False
        elif self.waypoint is not None and (new_x, new_y) == self.waypoint and visited == 0:
            reward = 50  # first time touching the checkpoint
            done = False
            new_visited = 1
        elif (new_x, new_y) == self.goal and (self.waypoint is None or visited == 1):
            reward = 100
            done = True
        else:
            reward = -1
            done = False

        self.state = (new_x, new_y, new_visited)
        return self.state, reward, done, {}

    def render(self):
        grid = np.full((self.size, self.size), '.', dtype=str)
        for ox, oy in self.obstacles:
            grid[ox][oy] = '#'
        if self.waypoint is not None:
            wx, wy = self.waypoint
            grid[wx][wy] = 'C'
        gx, gy = self.goal
        grid[gx][gy] = 'G'
        sx, sy, _ = self.state
        grid[sx][sy] = 'A'
        print('\n'.join(' '.join(row) for row in grid))
        print()