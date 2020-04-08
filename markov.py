#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 21:07:03 2020

@author: matthewbourque
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from state_space import StateSpace
from metrics import Dif1, Dif2, Price
from math import floor, ceil

'''
## data ##
prices = pd.read_csv('EURCAD_1 min_30 D.csv', header = 0, usecols = ['close'])
train_split = 0.8
train_data = prices.head(ceil(len(prices)*train_split))
test_data = prices.tail(floor(len(prices)*(1-train_split)))

# met1 
nbins1 = 2
mm1 = 1
ep1 = 0.1

# met2
nbins2 = 2
mm2 = 3
ep2 = 0.1

# declare metrics
met1 = Dif1(nbins1, mm1, ep1)
met2 = Dif2(nbins2, mm2, ep2)
metric_list = [met1,met2]
'''

class Markov():
    def __init__(self, metric_list, train_data):
        
        self.train_data = train_data
        self.metric_list = metric_list
        self.n_mets = len(self.metric_list)
                
        # state space
        self.X = StateSpace(self.metric_list, self.train_data)
        self.state_map = self.X.state_map
        self.inv_map = self.X.inverse_map
        self.h_map = self.X.h_state_map
        self.state_poss = self.X.state_poss_seq
        self.n_states = len(self.state_map)
        
        self.trans_probs = np.zeros((len(self.h_map), len(self.state_poss)))
        
        
        # initialize data
        self.markov_data = []
        self.markov_data.append(int(self.n_states*np.random.random()))
        
    def fit(self):
        self.state_seq = []
        print('Generating transition probabilities...')
        for i in range(self.X.pts_required, len(self.train_data)-self.X.pts_required):
            
            self.state_seq.append(self.X.get_state(self.train_data.iloc[i-self.X.pts_required: i]))
            state_tuple = self.inv_map[self.state_seq[-1]]
            
            h_seq = []
            s_seq = []
            ctr = 0
            for metric in self.X.metric_list:
                for j in range(len(metric.poss_h_seq)):
                    if metric.poss_seq[state_tuple[ctr]][0:metric.markov_mem] == metric.poss_h_seq[j]:
                        h_seq.append(j)
                
                s_seq.append(metric.poss_seq[state_tuple[ctr]][metric.markov_mem])
                ctr += 1
                
                if ctr == self.n_mets:
                    for k in range(len(self.state_poss)):
                        if tuple(s_seq) == self.state_poss[k]:
                            s_idx = k
            
                
            row = self.h_map[tuple(h_seq)]            
            self.trans_probs[row, s_idx] += 1
        
        row_cnt = []
        for i in range(len(self.trans_probs)):
            row_cnt.append(np.sum(self.trans_probs[i,:]))
        
        for i in range(len(self.trans_probs)):
            if row_cnt[i] != 0:
                self.trans_probs[i,:] = self.trans_probs[i,:]/row_cnt[i]
        
        print('done!')
        #return self.trans_probs
        
    def generate(self, data_len: int):
        """data_len = 93600 is 1Q worth of data"""
        
        print('generating markov data...')
        for i in range(data_len):
            
            state_tuple = self.inv_map[self.markov_data[-1]]
            
            seq = []
            h_seq = []
            s_seq = []
            ctr = 0
            for metric in self.X.metric_list:
                for j in range(len(metric.poss_h_seq)):
                    if metric.poss_seq[state_tuple[ctr]][-metric.markov_mem:] == metric.poss_h_seq[j]:
                        h_seq.append(j)
                        seq.append(metric.poss_h_seq[j])
                        
                s_seq.append(metric.poss_seq[state_tuple[ctr]][metric.markov_mem])
                ctr += 1
                            
            row = self.h_map[tuple(h_seq)]
            rand = np.random.random()
            
            probs = []
            for k in range(len(self.trans_probs[row,:])):
                probs.append(np.sum(self.trans_probs[row,:k]))
            
            s_idx = 0
            while rand >= probs[s_idx] and s_idx < len(probs)-1:
                s_idx +=1
                
            for l in range(self.n_mets):
                seq[l] += tuple([self.state_poss[s_idx][l]])
            
            state = []
            ctr = 0
            
            for metric in self.X.metric_list:
              state.append(metric.markov_dict[seq[ctr]])
              ctr +=1
                
            s = self.state_map[tuple(state)]
            
            self.markov_data.append(s)
            
        print('done!')
        return self.markov_data

'''                
markov = Markov(metric_list=metric_list, train_data=train_data)
markov.fit()
data = markov.generate(10)
'''