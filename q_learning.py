import numpy as np 
import random
import pickle

class Q_learn_agent:
    def __init__(self, shape, is_training=False, beta=None, gamma=None, epsilon=None) -> None:
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_training = is_training
        self.Q = np.zeros(shape)

    def dump_qfunction(self, path):
        with open(f"{path}", 'wb') as f:
            pickle.dump(self.Q, f)

    def pick_action(self, state):
        if random.random() < self.epsilon and self.is_training:
                return random.randint(0, 2)
        return np.argmax(self.Q[state])
    
    def learn(self, reward, prev_state, state, action):
        if self.is_training:
            update = reward + self.gamma * np.max(self.Q[prev_state]) \
                - self.Q[prev_state][action]
            self.Q[state][action] += self.beta * update
    
    def load_qfunction(self, path):
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)

