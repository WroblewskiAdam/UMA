import numpy as np 
import random
import pandas as pd
import pickle

class Q_learn_agent:
    def __init__(self, is_training=False, beta=None, gamma=None, epsilon=None) -> None:
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_training = is_training
        self.Q = np.zeros([11, 16, 3])

    def dump_qfunction(self, path):
        with open(f"{path}", 'wb') as f:
            pickle.dump(self.Q, f)

    def pick_action(self, state):
        state = self.discretization(state)
        if random.random() < self.epsilon and self.is_training:
                return random.randint(0, 2)
        return np.argmax(self.Q[state])
    
    def learn(self, reward, prev_state, state, action):
        state = self.discretization(state)
        prev_state = self.discretization(prev_state)
        if self.is_training:
            update = reward + self.gamma * np.max(self.Q[state]) - self.Q[prev_state][action]
            self.Q[state][action] += self.beta * update
    
    def load_qfunction(self, path):
        with open(path, 'rb') as f:
            self.Q = pickle.load(f)

    def discretization(self, state):
        discrete_percent = round(state[0]*10)
        bins = [-0.1,0.02,0.04,0.06,0.08,0.1,0.3,0.5,0.7,0.9,0.92,0.94,0.96,0.98,1,10,np.inf] #dyskretyzacja stanu
        labels = np.arange(0,len(bins)-1,1)
        binned_avg_distace = pd.cut(x=[state[1]], bins=bins, labels=labels)[0]
        return (discrete_percent, binned_avg_distace)
    
