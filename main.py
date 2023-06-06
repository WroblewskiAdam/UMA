from evolution_strategy import ES
from q_learning import Q_learn_agent
import pandas as pd
import matplotlib.pyplot as plt
from cec2017 import functions
import time
import numpy as np



gamma = 0.9
beta = 0.8
epsilon = 0.3

dimension = 10 # stała
k = 10
function = functions.f4
es_iterations = 1000 # stała
train_epochs = 100 # stała 


gamma = [2,4]
shape = [k+1, 16, 3]

def main():
    results = []
    for my_g in gamma:
        g_array = []
        for i in range(20):
            Q_ag = Q_learn_agent(gamma, beta, epsilon, shape, False)
            Q_ag.load_qfunction(f'model_g{my_g}_b5_e3_k10_f4_ver{i}')
            my_ES = ES(dimension=dimension, k=k, function=function, init_sigma=1)
            start_time = time.time()
            result = my_ES.es_rl(es_iterations, Q_ag)
            
            # my_ES.es_rl_training(num_epochs=train_epochs, num_iter=es_iterations , Q=Q_ag)
            end = time.time()
            print(end - start_time)
            # Q_ag.dump_qfunction('hist_model')
            # my_ES.es_standard(1000)
            g_array.append(result)
        results.append(g_array)

    print(results)
    print('gamma 0,1: ', np.average(results[0]))
    print('gamma 0,3: ', np.average(results[1]))




if __name__ == "__main__":
    main()
   

