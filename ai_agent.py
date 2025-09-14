import random
from config import ACTIONS, ALPHA, GAMMA, EPSILON

class QLearningAgent:
    def __init__(self):
        self.Q = {}

    def get_state_key(self, state):
        return tuple(int(v/10) for v in state)  # дискретизация

    def choose_action(self, state):
        key = self.get_state_key(state)
        if key not in self.Q:
            self.Q[key] = {a: 0 for a in ACTIONS}
        if random.random() < EPSILON:
            return random.choice(ACTIONS)
        return max(self.Q[key], key=self.Q[key].get)

    def learn(self, state, action, reward, next_state):
        key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)

        if key not in self.Q:
            self.Q[key] = {a: 0 for a in ACTIONS}
        if next_key not in self.Q:
            self.Q[next_key] = {a: 0 for a in ACTIONS}

        best_next = max(self.Q[next_key].values())
        self.Q[key][action] += ALPHA * (reward + GAMMA * best_next - self.Q[key][action])
