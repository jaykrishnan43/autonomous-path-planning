import numpy as np
import random

class QLearningAgentWaypoint:
    def __init__(self, grid_size, action_space, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.grid_size = grid_size
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Extra dimension vs. the original agent: visited-checkpoint flag (0 or 1)
        self.q_table = np.zeros((grid_size, grid_size, 2, action_space))

    def choose_action(self, state):
        x, y, visited = state
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            return int(np.argmax(self.q_table[x, y, visited]))

    def update(self, state, action, reward, next_state, done):
        x, y, visited = state
        next_x, next_y, next_visited = next_state

        current_q = self.q_table[x, y, visited, action]

        if done:
            target = reward
        else:
            best_next_q = np.max(self.q_table[next_x, next_y, next_visited])
            target = reward + self.gamma * best_next_q

        self.q_table[x, y, visited, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay