from evolution_strategy import ES
from qlearning import Q_learn_agent
import pandas as pd
import matplotlib.pyplot as plt
from cec2017 import functions


gamma =0.8
beta =0.8
epsilon = 0.3
shape = [100, 21, 3]

dimension = 10
k = 10
function = functions.f5
es_iterations = 10000
train_epochs =100

def main():
    Q = Q_learn_agent(gamma, beta, epsilon, shape, True)
    my_ES = ES(dimension=dimension, k=k, function=function, init_sigma=1)
    my_ES.es_rl_training(num_epochs=train_epochs, num_iter=es_iterations , Q=Q)

if __name__ == "__main__":
    main()
   

