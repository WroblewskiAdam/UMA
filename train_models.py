from evolution_strategy import ES, Reward
from q_learning import Q_learn_agent
import pandas as pd
from cec2017 import functions
import numpy as np


def main():
    testing_k = [5, 10 , 20, 50, 100]
    testing_beta = [0.1, 0.3, 0.5, 0.7, 0.9]
    testing_gamma = [0.1, 0.3, 0.5, 0.7, 0.9]
    testing_epsilon = [0.1, 0.3, 0.5, 0.7, 0.9]

    function = functions.f5
    dimension = 10

    k = 20
    iterations = 1000
    training_epochs = 200

    beta = 0.9
    gamma = 0.3
    epsilon = 0.5

    data = {
        "srednia" : [],
        "max" : [],
        "min" : [],
        "st_dev" : []
    }
    
    for k in testing_k:
        Q_ag = Q_learn_agent(is_training=True, beta=beta, gamma=gamma, epsilon=epsilon)
        my_ES = ES(dimension=dimension, k=k, function=function)
        my_ES.es_rl_training(max_epochs=training_epochs, max_iter=iterations, Q_ag=Q_ag, reward=Reward.LogDiff)
        file_name = f"model_g{int(gamma*10)}_b{int(beta*10)}_e{epsilon*10}_k{k}_f9.pickle" 
        Q_ag.dump_qfunction(file_name)
        result_list = []    
        for i in range(30):
            Q_ag.is_training = False
            my_ES = ES(dimension=dimension, k=k, function=function)
            result, x = my_ES.es_rl(iterations, Q_ag,reward=Reward.LogDiff)
            result_list.append(result)
        
        data["srednia"].append(np.mean(result_list))
        data["max"].append(*max(result_list))
        data["min"].append(*min(result_list))
        data["st_dev"].append(np.std(result_list))
    
    df = pd.DataFrame(data, index=testing_k)
    latex_code = df.to_latex(index=True, decimal=',', float_format="%.2f")
    latex_code = latex_code.replace("\\\n", "\\ \hline\n")
    latex_code = latex_code.replace(".", ",")
    print(latex_code)



if __name__ == "__main__":
    main()
   
