"""State space for Q-Learning agents."""
from functools import reduce
from metrics import Dif1, Dif2
from math import floor, ceil
import pandas as pd

class StateSpace():
    """States of the RL agent.
    'metrics' is a list of Metric type objects."""

    def __init__(self, metrics: list):

        if type(metrics) != "list": TypeError("metrics must be a list of Metric objects.")
        if not metrics: raise ValueError("metric list must be non-empty.")

        self.metrics = metrics

        # get properties of the state space
        if len(metrics) > 1:
            self.cardinality = reduce(lambda a,b : a.num_vals * b.num_vals, self.metrics)
        elif len(metrics) == 1:
            self.cardinality = metrics[0].num_vals
            
    def __str__(self): return "State space of size " + str(self.cardinality)

def main():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    # testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    dif1 = Dif1(1, 10, training_data)
    dif2 = Dif2(0, 10, training_data)

    X = StateSpace([dif1, dif2])
    print(X)

if __name__ == "__main__":
    main()
    # type_test()
