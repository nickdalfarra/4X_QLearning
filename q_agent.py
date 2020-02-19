"""
Created on Sun Feb  9 14:19:42 2020
Last edited by Nick Dal Farra Feb 18, 2020
@author: matthewbourque
"""
import numpy as np
import pandas as pd

from math import ceil, floor
from random import choice
from metrics import Price, Dif1, Dif2
from state_space import StateSpace
from action_space import ActionSpace

class QAgent():
    """Initialized with pre-built state spaces and action spaces."""
    def __init__(self, X: StateSpace, A: ActionSpace, discount_factor: float, high_init: bool = False):
        self.X = X
        self.A = A
        if high_init:
            self.q_table = np.full((self.X.cardinality, self.A.cardinality), 10)
        else:
            self.q_table = np.zeros((self.X.cardinality, self.A.cardinality))
        self.discount_factor = discount_factor
        self.visitations = np.ones((self.X.cardinality))
    
    def train(self, training_data: pd.DataFrame):
        """Train the Q-Table."""
        prev_s_idx = self.X.get_state(training_data.iloc[0 : self.X.pts_required])

        for n in range(self.X.pts_required+1,len(training_data)-1):
            """Reinforce prev_s_idx. n is current data index, so n-1 would be previous."""

            # get current state
            s_idx = self.X.get_state(training_data.iloc[n - self.X.pts_required : n])

            # set learning rate based on number of previous visitations
            alpha = 1/(self.visitations[prev_s_idx])

            # get new state from state space assuming states are iid
            prev_price = training_data.iloc[n-1].values[0]
            price = training_data.iloc[n].values[0]
            b = [1, price/prev_price]
            
            # reinforce all actions which could have been taken (relies on independence assumption between action taken and state evolution)
            current_max_q = np.max(self.q_table[s_idx, :])
            for a_idx in range(self.A.cardinality-1):
                r = np.log(np.dot(self.A.actions[a_idx], b))
                self.q_table[prev_s_idx, a_idx] += alpha*(r + self.discount_factor*current_max_q - self.q_table[prev_s_idx, a_idx])

            # add 1 to the number of visitations
            self.visitations[prev_s_idx] += 1

            prev_s_idx = s_idx

    def trade(self, testing_data: pd.DataFrame):
        """Excecute trades using argmax on the Q-table."""
        # get current state
        s_idx = self.X.get_state(testing_data.iloc[i-self.X.pts_required: i])

        # find all actions with maximum Q value at current state
        max_actions = np.argwhere(self.q_table[s_idx,:]==np.amax(self.q_table[s_idx,:]))

        # randomly select the index of a maximizing action
        a_idx = choice(max_actions) # "choice" is from the random package

        # get the action associate with the index
        action = self.A.actions[a_idx]

def main():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # declare metrics
    dif1 = Dif1(5, 0)

    # create state space
    X = StateSpace([dif1], training_data)
    
    # create action space
    A = ActionSpace(1, 5)

    # create Q-Learning agent
    gamma = 0.75 # discount factor (should this be zero? we assume transition prob is independent of action taken)
    agent = QAgent(X, A, gamma, high_init = False)

    agent.train(training_data)
    print(agent.q_table)
    print(agent.visitations)

if __name__ == "__main__":
    main()