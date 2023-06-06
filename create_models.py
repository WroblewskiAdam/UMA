from evolution_strategy import ES
from q_learning import Q_learn_agent
import pandas as pd
import matplotlib.pyplot as plt
from cec2017 import functions
import numpy as np


def main():
    testing_k = [5, 10, 20]
    testing_beta = [0.2, 0.4, 0.6, 0.8]
    testing_gamma = [0.2, 0.4, 0.6, 0.8]
    testing_epsilon = [0.2, 0.4, 0.6, 0.8]

    function = functions.f4
    dimension = 10 # stała

    k = 10
    iterations = 10000 # stała
    training_epochs = 200 # stała 
    shape = [k+1, 16, 3]

    beta = 0.6
    gamma = 0.4
    epsilon = 0.4

    data = {
        "srednia" : [],
        "max" : [],
        "min" : [],
        "st_dev" : []
    }
    
    for beta in testing_beta:
        Q_ag = Q_learn_agent(shape=shape, is_training=True, beta=beta, gamma=gamma, epsilon=epsilon)
        my_ES = ES(dimension=dimension, k=k, function=function, init_sigma=1)
        my_ES.es_rl_training(max_epochs=training_epochs, max_iter=iterations , Q_ag=Q_ag)
        file_name = f"model_g{int(gamma*10)}_b{int(beta*10)}_e{epsilon*10}_k{k}_f4.pickle" 
        Q_ag.dump_qfunction(file_name)#zapis do pliku

        result_list = []
        #wywołanie nauczonego algorytmu    
        for i in range(30):
            Q_ag.is_training = False
            my_ES = ES(dimension=dimension, k=k, function=function, init_sigma=1)
            result = my_ES.es_rl(iterations, Q_ag)
            result_list.append(result)
        
        data["srednia"].append(np.mean(result_list))
        data["max"].append(*max(result_list))
        data["min"].append(*min(result_list))
        data["st_dev"].append(np.std(result_list))
    
    df = pd.DataFrame(data, index=testing_gamma)
    print(df.to_latex(index=False, float_format="{:.2f}".format, decimal=',') )
    df.to_excel('beta_testing.xlsx')



if __name__ == "__main__":
    main()
   