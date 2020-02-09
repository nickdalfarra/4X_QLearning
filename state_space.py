"""State space for Q-Learning agents."""
from functools import reduce
from metrics import Dif1, Dif2
from math import floor, ceil
import pandas as pd

class StateSpace():
    """States of the RL agent.
    'metrics' is a list of Metric type objects."""

    def __init__(self, metric_list: list, training_data: pd.DataFrame):

        # check to ensure metric list is a non-empty list
        if type(metric_list) != "list": 
            raise TypeError("metrics must be a list of Metric objects.")
        if not metric_list: 
            raise ValueError("metric list must be non-empty.")
        self.metric_list = metric_list

        # fit the metrics to the training data

        # get properties of the state space
        if len(metric_list) > 1:
            self.cardinality = reduce(lambda a,b : a.metric_vals * b.metric_vals, self.metric_list)
        elif len(metric_list) == 1:
            self.cardinality = metric_list[0].metric_vals
            
    def __str__(self): return "State space of size " + str(self.cardinality)

def main():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    # testing_data = prices.tail(floor(len(prices)*(1-training_split)))


    dif1 = Dif1(1, 10)
    dif2 = Dif2(0, 10)
    X = StateSpace([dif1, dif2], training_data)

    print(X)

if __name__ == "__main__":
    main()
    # type_test()
