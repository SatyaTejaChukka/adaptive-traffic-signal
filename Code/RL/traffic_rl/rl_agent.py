import os
import pickle
import random


class QLearningAgent:
    def __init__(self, actions, alpha, gamma, epsilon, epsilon_min, epsilon_decay):
        self.actions = list(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def _ensure_state(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def act(self, state, training=True):
        q_values = self._ensure_state(state)
        if training and random.random() < self.epsilon:
            return random.choice(self.actions)
        max_q = max(q_values)
        best_actions = [action for action, value in zip(self.actions, q_values) if value == max_q]
        return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        current_q_values = self._ensure_state(state)
        next_q_values = self._ensure_state(next_state)
        action_index = self.actions.index(action)
        target = reward if done else reward + (self.gamma * max(next_q_values))
        current_q_values[action_index] += self.alpha * (target - current_q_values[action_index])
        return current_q_values[action_index]

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            if self.epsilon < self.epsilon_min:
                self.epsilon = self.epsilon_min

    def save(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "wb") as model_file:
            pickle.dump(
                {
                    "q_table": self.q_table,
                    "epsilon": self.epsilon,
                    "actions": self.actions,
                },
                model_file,
            )

    def load(self, path):
        if not os.path.exists(path):
            return False
        with open(path, "rb") as model_file:
            payload = pickle.load(model_file)
        self.q_table = payload.get("q_table", {})
        self.epsilon = payload.get("epsilon", self.epsilon)
        saved_actions = payload.get("actions")
        if saved_actions:
            self.actions = list(saved_actions)
        return True
    