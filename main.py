from evolution_strategy import ES, Reward
from q_learning import Q_learn_agent
import pandas as pd
import matplotlib.pyplot as plt
from cec2017 import functions
import time
import numpy as np



gamma = 0.5
beta = 0.5
epsilon = 0.3

dimension = 10 # stała
k = 20
function = functions.f6
es_iterations = 1000 # stała
train_epochs = 200 # stała 


# gamma = [2,4]
shape = [11, 16, 3]

data = {
    "srednia" : [],
    "max" : [],
    "min" : [],
    "st_dev" : []
}


def main():
    results = []
    # for my_g in gamma:
    g_array = []
    # Q_ag = Q_learn_agent(gamma= gamma,beta= beta,epsilon= epsilon, shape=shape, is_training= True)
    # my_ES = ES(dimension=dimension, k=k, function=function)
    # my_ES.es_rl_training(max_epochs=train_epochs, max_iter=es_iterations, Q_ag=Q_ag, reward=Reward.LogDiff)
    # Q_ag.dump_qfunction('f7alt')
    for i in range(30):
        Q_ag = Q_learn_agent(gamma= gamma,beta= beta,epsilon= epsilon, shape=shape, is_training= False)
        Q_ag.load_qfunction(f'model_g5_b5_e3.0_k20_f6alt.pickle')
        my_ES = ES(dimension=dimension, k=k, function=function)
        result = my_ES.es_rl(es_iterations, Q_ag,reward=Reward.Percent)
        g_array.append(result)
    for _ in range(30):
        my_ES = ES(dimension=dimension, k=k, function=function)
        st_resoult = my_ES.es_standard(1000)
        results.append(st_resoult)
    data["srednia"].append(np.mean(g_array))
    data["max"].append(*max(g_array))
    data["min"].append(*min(g_array))
    data["st_dev"].append(np.std(g_array))
    data["srednia"].append(np.mean(results))
    data["max"].append(*max(results))
    data["min"].append(*min(results))
    data["st_dev"].append(np.std(results))


    df = pd.DataFrame(data, index=['qlearn', 'standard'])
    latex_code = df.to_latex(index=True, decimal=',', float_format="%.2f")
    latex_code = latex_code.replace("\\\n", "\\ \hline\n")
    latex_code = latex_code.replace(".", ",")
    print(latex_code)




if __name__ == "__main__":
    main()
   

