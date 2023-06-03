from evolution_strategy import ES
from qlearning import Q_learn_agent
import pandas as pd
import matplotlib.pyplot as plt
from cec2017 import functions


gamma = [0.9]
beta = [0.2, 0.4, 0.6, 0.8]
epsilon = [0.2, 0.4, 0.6, 0.8]

# gamma = 0.9
my_beta = 0.8
my_epsilon = 0.3

dimension = 10 # stała
k = 20
function = functions.f4
es_iterations = 10000 # stała
train_epochs = 100 # stała 
shape = [k+1, 16, 3]

def main():
    for my_gamma in gamma:
        Q_ag = Q_learn_agent(my_gamma, my_beta, my_epsilon, shape, True)
        my_ES = ES(dimension=dimension, k=k, function=function, init_sigma=1)
        my_ES.es_rl_training(num_epochs=train_epochs, num_iter=es_iterations , Q=Q_ag)
        
        file_name = f"model_g{int(my_gamma*10)}_b{int(my_beta*10)}_e{int(my_epsilon*10)}_k{k}"
        Q_ag.dump_qfunction(file_name)



if __name__ == "__main__":
    main()
   