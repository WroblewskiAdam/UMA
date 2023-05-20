import numpy as np 
import random
from cec2017.functions import f4 as func
import matplotlib.pyplot as plt

percent_number = 100
dim_state_number = 10
action_number = 3

def q_func_inicialize(shape):
    return np.zeros(shape)

def train(gamma, beta, reward, state, epsilon, Q, is_terminal, es1_1):
    if random.random() < epsilon:
        new_action = random.randint(0, 3)
    else:
        new_action = np.argmax(Q[state])
    
    
        
    if not is_terminal:
        update = reward + gamma * np.max(Q[state]) - Q[state][new_action]
    else:
        update = reward - Q[state][new_action]
    Q[self.obs][self.action] += reward * update



q_func = q_func_inicialize([percent_number, dim_state_number, action_number])

# print(func([[-50,100]]))

# print(q_func)

