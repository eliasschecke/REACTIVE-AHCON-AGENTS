from Game import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import GradientTape, expand_dims
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Auswertung:
    def __init__(self, agent_list = None, rounds = 100):
        self.agent_list=agent_list
        self.rounds = rounds
        self.real_reinforcement = []
        self.lifetime = []
        self.food_found = []
        self.energy = []
        
    def run(self):
        self.real_reinforcement = []
        self.lifetime = []
        self.food_found = []
        self.energy = []
        for agent in self.agent_list:
            rr, lt, ff, e, _ = agent.analyse_run(rounds=self.rounds)
            self.real_reinforcement.append(rr)
            self.lifetime.append(lt)
            self.food_found.append(ff)
            self.energy.append(e)
        return
    
    def food_distribution_vis(self):
        data_list = []
        mean = []
        for ff in self.food_found:
            unique, counts = np.unique(ff, return_counts=True)
            amount = counts.sum()
            food_list=[]
            for j in range(16):
                food_list.append(counts[unique==j].sum()/amount)
            data_list.append(food_list)
            mean.append(ff.mean())
    
        data=pd.DataFrame(np.array(data_list),columns=list(range(16)))
        data.index = mean
        return data.plot(kind='bar', stacked=True, colormap = "tab20").legend(loc='center left', bbox_to_anchor=(1, 0.5))
    

            

            