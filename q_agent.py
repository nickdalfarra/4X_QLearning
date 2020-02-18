"""
Created on Sun Feb  9 14:19:42 2020
Last edited by Nick Dal Farra Feb 18, 2020
@author: matthewbourque
"""
import numpy as np
import pandas as pd

from math import ceil, floor
from metrics import Price, Dif1, Dif2
from state_space import StateSpace
from action_space import ActionSpace

# from action_space import *
# from state_space import *

class QAgent():
    def __init__(self, training_data: pd.DataFrame):
        pass

def main():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # declare metrics
    dif1 = Dif1(10, 0)       # markov memory of 1 (STILL YET TO BE IMPLEMENTED), 5 bins

    # creating state and action spaces
    X = StateSpace([dif1], training_data)
    s_dict  = X.state_map

    # generates state space
    s_space = []
    for key in s_dict:
        s_space.append(s_dict[key])

    # get current state
    for i in range(X.pts_required,len(training_data)-X.pts_required):
        state = X.get_state(training_data.iloc[i-X.pts_required: i])
    
    # generate action space   
    num_ccy = 1
    precision = 1000

    A = ActionSpace(num_ccy, precision)
    a_space = A.actions

    #print('state space:\n', s_space)
    #print('action space:\n', a_space)


    ## Updating Q Table ##
    n_states = len(s_space)-1
    n_actions = len(a_space)-1

    # initialize q table with zeros
    q_table = np.zeros((n_states, n_actions))

    # initilize learning rate table with ones
    lr_table = np.ones((n_states, n_actions))

    # discount factor (how much do we care about reward now vs reward later)
    gamma = 0.75


    # loop through training data
    for i in range(X.pts_required,len(training_data)-1):
        # initialize state
        s_idx = X.get_state(training_data.iloc[i-X.pts_required: i])
        
        # if all actions for a state have zero reward 0 randomly select action
        if np.sum(q_table[s_idx,:]) == 0:
            # randint randomly selects ints from 0 up to but not including n_actions
            a_idx = np.random.randint(0, n_actions)

        # select the action with largest reward value in state s
        else:
            a_idx = np.argmax(q_table[s_idx, :])

        # get action from action space
        a = a_space[a_idx]

        # get new state from state space assuming states are iid
        price = training_data.iloc[i].values[0]
        next_price = training_data.iloc[i+1].values[0]
        b = [1, next_price/price]
        
        # calculate reward
        r = np.log(np.dot(a, b)) 
        
        # choose learning rate
        if lr_table[s_idx, a_idx] == 0:
            alpha = 1
        else:
            alpha = 1/(lr_table[s_idx, a_idx])

        # update q table for action a taken on state s
        next_s_idx = X.get_state(training_data.iloc[i+1-X.pts_required: i+1])
        q_table[s_idx, a_idx] += r + alpha*(gamma*np.max(q_table[next_s_idx, :]) - q_table[s_idx, a_idx])
        
        # update learning rate
        lr_table[s_idx, a_idx] += 1

    print(q_table)

if __name__ == "__main__":
    main()