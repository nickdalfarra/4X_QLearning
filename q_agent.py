#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 19:55:41 2020

@author: matthewbourque
"""

import numpy as np
import pandas as pd

from action_space import ActionSpace
from state_space import StateSpace
from markov import Markov
from metrics import Dif1, Dif2, Price
from math import floor, ceil


## data ##
prices = pd.read_csv('EURCAD_1-min_30-D.csv', header = 0, usecols = ['close'])
train_data = pd.read_csv('USDCAD_1min_2019.csv', header = 0, usecols = ['close'])
test_data = pd.read_csv('USDCAD_M1_2020.csv', header = 0, usecols = ['close'])
# train_split = 0.8
# train_data = prices.head(ceil(len(prices)*train_split))
# test_data = prices.tail(floor(len(prices)*(1-train_split)))

# met1 
nbins1 = 2
mm1 = 0
ep1 = 0.1

# met2
nbins2 = 2
mm2 = 1
ep2 = 0.1

#met3
nbins3 = 5
mm3 = 3
ep3 = 0.1

# declare metrics
met1 = Dif1(nbins1, mm1, ep1)
met2 = Dif2(nbins2, mm2, ep2)
met3 = Price(nbins3, mm3, ep3)
metric_list = [met1, met3]
n_mets = len(metric_list)

class Qagent():
    def __init__(self, num_ccy: int, precision: int, gamma: float, metric_list: list, train_data: pd.DataFrame):
        
        self.num_ccy = num_ccy
        self.precision = precision
        self.gamma = gamma
        self.metric_list = metric_list
        self.train_data = train_data

        self.A = ActionSpace(self.num_ccy, self.precision)
        self.a_space = self.A.actions
        self.X = StateSpace(self.metric_list, self.train_data)
        self.state_map = self.X.state_map
        
        self.n_states = len(self.state_map)
        self.n_actions = len(self.a_space)
        
        self.q_table = np.zeros((self.n_states, self.n_actions))
        self.lr_table = np.zeros((self.n_states, self.n_actions))
        
    def train(self, data=0, markov=False):

        if markov == True:
            train_start = 0
            train_end = len(data)-1
        else:
            train_start = self.X.pts_required
            train_end = len(self.train_data)-1
            
        print('Training q_table...')        
        for i in range(train_start, train_end):
            
            # initialize state
            if markov == True:
                s_idx = data[i]         
            else:
                s_idx = self.X.get_state(self.train_data.iloc[i-self.X.pts_required: i])
            
            
            # if all actions for a state have zero reward 0 randomly select action
            if np.sum(self.q_table[s_idx,:]) == 0:
                # randint randomly selects ints from 0 up to but not including n_actions
                a_idx = np.random.randint(0, self.n_actions)
        
            # select the action with largest reward value in state s
            else:
                a_idx = np.argmax(self.q_table[s_idx, :])
        
            # get action from action space
            a = self.a_space[a_idx]
        
            # get new state from state space assuming states are iid
            price = self.train_data.iloc[i - 1].values[0]
            next_price = self.train_data.iloc[i].values[0]
            b = [1, next_price/price]
            z = -1
            # calculate reward
            r = np.log(np.dot(a, b))
            #print(r)
            
            # choose learning rate
            if self.lr_table[s_idx, a_idx] == 0:
              alpha = 1
            else:
              alpha = 1/(self.lr_table[s_idx, a_idx])
        
            # update q table for action a taken on state s
            if markov == True:
                next_s_idx = data[i+1]
            else:
                next_s_idx = self.X.get_state(train_data.iloc[i+1-self.X.pts_required: i+1])
            
            #print(next_s_idx)
            self.q_table[s_idx, a_idx] += r + alpha*(self.gamma*np.max(self.q_table[next_s_idx, :]) - self.q_table[s_idx, a_idx])
            
            # update learning rate
            self.lr_table[s_idx, a_idx] += 1

        print('done!')
        return self.q_table, self.lr_table
        

q_agent = Qagent(num_ccy=1, precision=1000, gamma=0.75, metric_list=metric_list, train_data=train_data)
#markov = Markov(metric_list=metric_list, train_data=train_data)
#markov.fit()
#data = markov.generate(1000)
q_table, lr_table = q_agent.train()
df = pd.DataFrame(q_table)
df.to_csv('q_table5.csv')