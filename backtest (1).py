#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 10:34:04 2020

@author: matthewbourque
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stat
from action_space import ActionSpace
from state_space import StateSpace
from markov import Markov
from metrics import Dif1, Dif2, Price
from math import floor, ceil
from market import Market

## data ##
prices = pd.read_csv('EURCAD_1-min_30-D.csv', header = 0, usecols = ['close'])
train_split = 0.2
train_data = prices.head(ceil(len(prices)*train_split))
test_data = prices.tail(floor(len(prices)*(1-train_split)))
rain_data = pd.read_csv('USDCAD_1min_2019.csv', header = 0, usecols = ['close'])
test_data = pd.read_csv('USDCAD_M1_2020.csv', header = 0, usecols = ['close'])
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
mm3 = 2
ep3 = 0.1

# declare metrics
met1 = Dif1(nbins1, mm1, ep1)
met2 = Dif2(nbins2, mm2, ep2)
met3 = Price(nbins3, mm3, ep3)
metric_list = [met2, met3]
n_mets = len(metric_list)

q_table = pd.read_csv('q_table4.csv')
q_table = np.array(q_table)
q_table = np.delete(q_table, 0, 1)



class Backtest():
    def __init__(self, q_table, metric_list, test_data, market):
        
        self.mkt = market
        self.q_table = q_table
        self.test_data = test_data
        self.num_ccy = 1
        self.metric_list = metric_list
        self.pnl = []
        self.time = []
        self.curr_pos = 0
        
        self.trades = []
    


        self.A = ActionSpace(self.num_ccy, self.q_table.shape[1]-1)
        self.a_space = self.A.actions
        self.X = StateSpace(self.metric_list, self.test_data)
        self.state_map = self.X.state_map
        
        self.n_states = len(self.state_map)
        self.n_actions = len(self.a_space)
        print('backtest initialized!')
        
    def update_pnl(self, price, curr_pos, i):
        if curr_pos == 1:
            self.pnl.append(np.log(price[i+1]/price[i]))
        elif curr_pos == -1:
          self.pnl.append(np.log(price[i]/price[i+1]))
        else:
          self.pnl.append(0)
    
    def plot(self):
        plt.rcParams["figure.figsize"] = (50,10) # (width, height)
        plt.plot(self.mkt.bought_time, self.mkt.bought_price, 'bo', label='buy')
        plt.plot(self.mkt.sold_time, self.mkt.sold_price, 'ro', label='sell')
        plt.plot(self.time, self.test_data, label='price')
        plt.legend(prop={'size': 20})
        plt.xlabel('Time', fontsize='x-large')
        plt.ylabel('Exchange Rate', fontsize='x-large')
        plt.show()
    
    def run(self, data=0, markov=False):
        if markov == True:
            test_start = 0
            test_end = len(data)-1
        else:
            test_start = self.X.pts_required
            test_end = len(self.test_data)-1
        
        self.time = [t for t in range(test_end+1)]
        
        # generate random initial action
        prev_action = self.a_space[int(self.n_actions*np.random.random())][1]
        
        print('backtesting over backtest length of: ', test_end)
        for i in range(test_start, test_end):

            if markov == True:
                s_idx = data[i]
            else:
                s_idx = self.X.get_state(self.test_data.iloc[i-self.X.pts_required: i])

            next_action = self.a_space[np.argmax(self.q_table[s_idx, :])][1]
            self.trades.append(next_action - prev_action)
            
            self.curr_pos = self.mkt.update_pos(next_action=next_action, prev_action=prev_action, price=np.array(test_data.iloc[i]), price_idx=i)
            self.update_pnl(np.array(test_data), self.curr_pos, i)
            
            prev_action = next_action
        bi_weekly = []
        x = np.array(self.pnl, dtype=float)
        y = list(x)
        bi_weekly.append(sum(y[0:15826]))
        bi_weekly.append(sum(y[15827:31653]))
        print('trades per hour:',int(bt.mkt.trade_cnt*60/len(bt.time)))
        print('returns:',np.sum(self.pnl)[0]*100)
        print('Volatility:',stat.stdev(bi_weekly))

bt = Backtest(q_table=q_table, metric_list=metric_list, test_data=test_data, market=Market())
bt.run()
bt.plot()
    
    