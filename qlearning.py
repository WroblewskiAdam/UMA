import numpy as np 
import random
from cec2017.functions import f4 as func
# import matplotlib.pyplot as plt
import pickle
# from uma_projekt import ES
import pandas as pd

percent_number = 100
dim_state_number = 21
action_number = 3

shape = [100, 21, 3]

class Q_learn_agent:
    def __init__(self, gamma, beta, epsilon, shape, is_training, ES: ES) -> None:
        self.gamma = gamma
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.is_training = is_training
        self.Q = self.q_func_inicialize(shape)
        self.ES = ES

    def q_func_inicialize(self, shape):
        return np.zeros(shape)

    def dump_qfunction(self):
        with open(f"model/qfunc.pickle", 'wb') as f:
            pickle.dump(self.Q, f)

    def pick_action(self, state):
        if random.random() < self.epsilon and self.is_training:
                return random.randint(0, 3)
        return np.argmax(self.Q[state])
    
    def learn(self, reward, prev_state, state, action):
        update = reward + self.gamma * np.max(self.Q[prev_state]) \
            - self.Q[prev_state][action]
        self.Q[state][action] += reward * update
    
    def load_qfunction(self, path):
        with open(path, 'rb') as f:
            self.Q = pickle.load()




# q_func = q_func_inicialize([percent_number, dim_state_number, action_number])

# print(func([[-50,100]]))

# print(q_func)

