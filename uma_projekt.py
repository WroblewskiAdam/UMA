import numpy as np
import pandas as pd
from cec2017.functions import f4 as func
from qlearning import Q_learn_agent

class ES:
    def __init__(self, dimension, k, init_sigma=1):
        self.dimension = dimension
        self.x = self.init_population()
        self.y = None
        self.score_x = None
        self.score_y = None
        self.success_mem = []
        self.sigma = init_sigma
        self.iteration = 0
        self.k = k
        self.past_population = []



    def init_population(self, samples = 1):
        # Inicjalizacja osobnika do ES 1+1
        # np.random.seed(10)
        x = np.random.uniform(-50, 50, size=(samples, self.dimension))
        return x


    def evaluate(self, x):
        val = func(x)
        return val


    def get_mutant(self, x):
        y = x + self.sigma * np.random.normal(loc=0.0, scale=1.0, size=self.dimension)
        return y


    def es_standard(self, max_iter, k):
        self.iteration = 0

        while(self.iteration < max_iter):
            self.y = self.get_mutant(self.x)
            self.past_population.append(self.x)

            self.score_x = self.evaluate(self.x)
            self.score_y = self.evaluate(self.y)
            if self.score_y < self.score_x:
                self.success_mem.append(True)
                self.x = self.y
            else:
                self.success_mem.append(False)

            if self.iteration % k == 0:
                avg_distance = self.calc_distance()
                bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,20,30,40,50,60,70,80,90,100,np.inf]
                labels = np.arange(0,len(bins)-1,1)
                binned_avg_distace = pd.cut(x=[avg_distance], bins=bins, labels=labels)[0]
                print(binned_avg_distace)
                if sum(self.success_mem)/len(self.success_mem) >= 0.2:
                    self.sigma = 1.22*self.sigma
                elif sum(self.success_mem)/len(self.success_mem) < 0.2:
                    self.sigma = 0.82*self.sigma
                self.success_mem = []
                self.past_population = []
            # print((self.iteration, self.score_x[0]))
            self.iteration += 1


        return (self.x, self.score_x)


    def es_rl_training(self, num_epochs, num_iter, k, Q):
        self.iteration = 0
        for _ in range(num_epochs):
            self.x = self.init_population()
            self.es_rl(num_iter, k, Q)

    def es_rl(self, num_iter, k, Q: Q_learn_agent):
        state = ()
        action = 0
        while(self.iteration < num_iter):
            self.past_population.append(self.x)
            self.y = self.get_mutant(self.x)
            self.score_x = self.evaluate(self.x)
            self.score_y = self.evaluate(self.y)
            if self.score_y < self.score_x:
                self.succes_mem.append(True)
                self.x = self.y
            else:
                self.succes_mem.append(False)

            if self.iteration % k == 0:
                success_percent = round(sum(self.succes_mem)/len(self.succes_mem))
                avg_distance = self.calc_distance()
                bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,20,30,40,50,60,70,80,90,100,np.inf]
                labels = np.arange(0,len(bins)-1,1)
                binned_avg_distace = pd.cut(x=[avg_distance], bins=bins, labels=labels)[0]
                action = Q.pick_action(state) if self.iteration!=0 else 1
                match action:
                    case 0:
                        self.sigma = 0.82*self.sigma
                    case 2:
                        self.sigma = 1.22*self.sigma
                    case _:
                        pass
                self.succes_mem = []
                self.iteration = 0
                prev_state = state
                state = (success_percent, binned_avg_distace)
                if self.iteration!=0:
                    Q.learn(success_percent, prev_state, state, action)


    def calc_distance(self):
        mass_center = np.array(self.past_population).mean(0)
        distance = 0
        for x in self.past_population:
            distance += np.linalg.norm(mass_center - x)
        return distance
        

def main():
    my_ES = ES(50,10)
    my_ES.es_standard(10000, k=10)

    # bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,10,20,30,40,50,60,70,80,90,100,np.inf]
    # labels = np.arange(0,len(bins)-1,1)
    # binned = pd.cut(x=[200], bins=bins, labels=labels)
    # print(binned[0])

if __name__ == "__main__":
    main()
   
