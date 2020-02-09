"""State space for Q-Learning agents."""
from functools import reduce
from itertools import product
from metrics import Dif1, Dif2
from math import floor, ceil
import pandas as pd

class StateSpace():
    """States of the RL agent.
    'metrics' is a list of Metric type objects."""

    def __init__(self, metric_list: list, training_data: pd.DataFrame):
        """Automatically fit metrics and calculate stochastic kernel."""

        # check to ensure metric list is a non-empty list
        if not isinstance(metric_list, list): 
            raise TypeError("metric_list must be a list of Metric objects.")
        if not metric_list:
            raise ValueError("metric_list must be non-empty list.")
        self.metric_list = metric_list

        # fit the metrics to the training data
        for metric in self.metric_list:
            metric.fit(training_data)

        # make a list of all possible states, and enumerate them to map them to Q-Table values
        self.states = [*product(*[range(metric.n_vals) for metric in self.metric_list])]
        backward_dict = dict(enumerate(self.states)) 
        self.state_map = {y:x for x,y in backward_dict.items()} # input state

    def get_state(self, data: pd.DataFrame):
        """Calculate the state of the data at the bottom of a DataFrame."""

        metric_values = []
        for metric in self.metric_list:
            metric_values.append(metric.get_val(data))

        return self.state_map[tuple(metric_values)]
            
    def __str__(self): return "State space with " + str(len(self.states)) + " states."

def main():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # declare metrics
    dif1 = Dif1(0, 5)       # markov memory of 1 (STILL YET TO BE IMPLEMENTED), 5 bins
    dif2 = Dif2(0, 3)       # markov memory of 0 (STILL YET TO BE IMPLEMENTED), 3 bins

    X = StateSpace([dif1, dif2], training_data)
    # print(X)

    # print a bunch of states
    start = -400
    for point in range(start,start+25):
        print(X.get_state(testing_data.iloc[point-3: point]) , end=', ')
    print(" ")

if __name__ == "__main__":
    main()
