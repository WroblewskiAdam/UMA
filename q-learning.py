import numpy as np 
import random
from cec2017.functions import f4 as func
# import matplotlib.pyplot as plt
import pickle
from uma_projekt import ES
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

    def train(self, state, num_epochs, num_iterations):
        for _ in range(num_epochs):
            new_action = 1
            for i in range(num_iterations):
                if random.random() < self.epsilon and i!=0:
                    new_action = random.randint(0, 3)
                elif i!=0:
                    new_action = np.argmax(self.Q[state])
                
                new_state, reward = self.ES.es_rl(new_action)

                update = reward + self.gamma * np.max(self.Q[state]) \
                    - self.Q[state][new_action]
                self.Q[new_state][new_action] += reward * update
                state = new_state
            else:
                update = reward - self.Q[state][new_action]
                self.Q[new_state][new_action] += reward * update
                state = new_state
    
    def run(self, num_iterations):
        new_action = 1
        for _ in range(num_iterations):
            new_state, _ = self.ES.es_rl(new_action)
            new_action = np.argmax(self.Q[new_state])


# q_func = q_func_inicialize([percent_number, dim_state_number, action_number])

# print(func([[-50,100]]))

# print(q_func)

