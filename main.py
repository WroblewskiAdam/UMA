from evolution_strategy import ES
from qlearning import Q_learn_agent
import pandas as pd
import matplotlib.pyplot as plt
from cec2017 import functions


gamma = 0.9
beta = 0.8
epsilon = 0.3

dimension = 10
k = 10
function = functions.f4
es_iterations = 10000
train_epochs = 100

shape = [round(100/k)+1, 21, 3]

def main():
    Q_ag = Q_learn_agent(gamma, beta, epsilon, shape, True)
    # Q_ag.load_qfunction('model2')
    my_ES = ES(dimension=dimension, k=k, function=function, init_sigma=1)
    # my_ES.es_rl(es_iterations, Q_ag)
    my_ES.es_rl_training(num_epochs=train_epochs, num_iter=es_iterations , Q=Q_ag)
    # Q_ag.dump_qfunction('model3')
    # my_ES.es_standard(10000)
if __name__ == "__main__":
    main()
   

