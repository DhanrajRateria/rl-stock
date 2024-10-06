# model.py

import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros([state_size, action_size])
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def choose_action(self, state):
        """Chooses action based on epsilon-greedy policy."""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state):
        """Updates the Q-table based on the Bellman equation."""
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        self.q_table[state, action] += self.learning_rate * (td_target - self.q_table[state, action])

    def train(self, environment, episodes=1000):
        """Train the agent by interacting with the environment."""
        for episode in range(episodes):
            state = environment.reset()
            done = False

            while not done:
                action = self.choose_action(state[0])  # Only use price index as state
                next_state, reward, done = environment.step(action)
                self.update_q_table(state[0], action, reward, next_state[0])
                state = next_state