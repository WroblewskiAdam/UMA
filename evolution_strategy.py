import numpy as np
import pandas as pd
from cec2017 import functions
from q_learning import Q_learn_agent
from matplotlib import pyplot as plt


class ES:
    def __init__(self, dimension, k, function, init_sigma=1):
        self.dimension = dimension
        self.x = None
        self.y = None
        self.score_x = None
        self.score_y = None
        self.success_mem = []
        self.sigma = init_sigma
        self.iteration = 0
        self.k = k
        self.past_population = []
        self.function = function
        self.training_epoch = 0

    def init_population(self, samples = 1):# Inicjalizacja osobnika początkowego
        x = np.random.uniform(-50, 50, size=(samples, self.dimension))
        self.x = x
        return x

    def evaluate(self, x):
        val = self.function(x)
        return val

    def get_mutant(self, x):
        y = x + self.sigma * np.random.normal(loc=0.0, scale=1.0, size=self.dimension)
        # print(y)
        return y

    def es_standard(self, max_iter):
        self.init_population()
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

            if self.iteration % self.k == 0:
                if sum(self.success_mem)/len(self.success_mem) >= 0.2:
                    self.sigma = 1.22*self.sigma
                elif sum(self.success_mem)/len(self.success_mem) < 0.2:
                    self.sigma = 0.82*self.sigma
                self.success_mem = []
                self.past_population = []
            self.iteration += 1
            print(self.evaluate(self.x))
        return (self.x, self.score_x)


    def es_rl_training(self, max_epochs, max_iter, Q_ag: Q_learn_agent):
        # distance_array = list()
        for _ in range(max_epochs):
            self.iteration = 0
            self.sigma = 1
            self.x = self.init_population()
            print("Epoka: ", _)
            array = self.es_rl(max_iter, Q_ag)
            self.training_epoch += 1
            # distance_array.extend(array)



    def es_rl(self, max_iter, Q: Q_learn_agent):
        self.init_population()
        distance_array = []
        state = ()
        action = 0
        while(self.iteration < max_iter):
            self.past_population.append(self.x)
            self.y = self.get_mutant(self.x)
            self.score_x = self.evaluate(self.x)
            self.score_y = self.evaluate(self.y)
            if self.score_y < self.score_x:
                self.success_mem.append(True)
                self.x = self.y
            else:
                self.success_mem.append(False)

            if self.iteration % self.k == 0:
                success_percent = round(sum(self.success_mem)/len(self.success_mem)*self.k)
                avg_distance = self.calc_distance()
                distance_array.append(avg_distance)
                action = Q.pick_action(state) if self.iteration!=0 else 0
                match action:
                    case 1:
                        self.sigma = 0.82*self.sigma
                    case 2:
                        self.sigma = 1.22*self.sigma
                    case _:
                        pass
                self.success_mem = []
                prev_state = state
                state = (success_percent, avg_distance)
                self.past_population = []
                self.success_mem = []
                if self.iteration!=0:
                    Q.learn(success_percent, prev_state, state, action)

            self.iteration += 1
            # print("Iter: %d, fcel: %d"%(self.iteration, self.evaluate(self.x)))
            print("Epoka uczenia: ", self.training_epoch, "Iter: ", self.iteration, " fcel: ", self.evaluate(self.x))
        # self.hist(distance_array)
        return self.score_x
            
            
    def calc_distance(self):
        mass_center = np.array(self.past_population).mean(0)
        distance = 0
        for x in self.past_population:
            distance += np.linalg.norm(mass_center - x)
        distance = distance/self.k
        bins = [-0.1,0.02,0.04,0.06,0.08,0.1,0.3,0.5,0.7,0.9,0.92,0.94,0.96,0.98,1,10,np.inf] #dyskretyzacja stanu
        labels = np.arange(0,len(bins)-1,1)
        binned_avg_distace = pd.cut(x=[distance], bins=bins, labels=labels)[0]
        return binned_avg_distace
    
    # def hist(self, array):
    #     plt.hist(array, bins='auto')
    #     plt.show()

