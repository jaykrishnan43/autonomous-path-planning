import numpy as np
import random

class QLearningAgent:
    def __init__(self, grid_size, action_space, alpha=0.1, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.grid_size = grid_size
        self.action_space = action_space

        # Learning rate: how much new info overrides old info
        self.alpha = alpha

        # Discount factor: how much future rewards matter vs immediate ones
        self.gamma = gamma

        # Exploration settings
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: one row per grid cell, one column per action
        # Shape: (size, size, num_actions)
        self.q_table = np.zeros((grid_size, grid_size, action_space))

    def choose_action(self, state):
        # Epsilon-greedy: sometimes explore randomly, sometimes exploit best known action
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_space - 1)
        else:
            x, y = state
            return int(np.argmax(self.q_table[x, y]))

    def update(self, state, action, reward, next_state, done):
        x, y = state
        next_x, next_y = next_state

        current_q = self.q_table[x, y, action]

        if done:
            target = reward  # no future reward if episode ended
        else:
            best_next_q = np.max(self.q_table[next_x, next_y])
            target = reward + self.gamma * best_next_q

        # Bellman update rule
        self.q_table[x, y, action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay