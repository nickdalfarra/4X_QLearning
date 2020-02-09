"""Metrics usable by the worker. Metrics must be given training data at initialization."""

import pandas as pd
from math import floor, ceil

class Metric():
    """Base metric class."""
    def __init__(self, pts_required: int, nbins: int, markov_mem: int):
        self.pts_required = pts_required
        self.nbins = nbins
        self.markov_mem = markov_mem  # Markov memory will influence model fitting in State_Space, not here
        self.metric_vals = nbins**markov_mem

class Dif1(Metric):
    """Bins on the first difference distribution."""
    def __init__(self, markov_mem: int, nbins: int, training_data: pd.DataFrame):
        pts_required = 2 # points required for the metric to be calculated
        super().__init__(pts_required, nbins, markov_mem)
        self.partitions = self.get_partitions(training_data)

    def get_partitions(self, training_data: pd.DataFrame):
        """Creates the partitions which define the metric mapping."""
        differences = training_data.diff()
        quantiles = [(x+1)/self.nbins for x in [*range(self.nbins-1)]]
        partitions = differences.quantile(quantiles)['Price'].tolist()
        return partitions

    def get_val(self, data: pd.DataFrame):
        """Get the metric's latest value on the data given."""
        # get the most recent first difference in the left-most column
        dif = data.tail(2).diff()[data.columns.values[0]].iloc[-1]

        # return the number of partitions to the left of the value
        val = 0
        while dif >= self.partitions[val] and val < len(self.partitions)-1: 
            val += 1
        if dif >= self.partitions[len(self.partitions)-1]: # case where dif is greater than all partition values
            val += 1 

        return val

class Dif2(Metric):
    """Bins on the second difference distribution."""
    def __init__(self, markov_mem: int, nbins: int, training_data: pd.DataFrame):
        pts_required = 3 # points required for the metric to be calculated
        super().__init__(pts_required, nbins, markov_mem)
        self.partitions = self.get_partitions(training_data)

    def get_partitions(self, training_data: pd.DataFrame):
        """Creates the partitions which define the metric mapping."""
        second_difs = training_data.diff().diff()
        quantiles = [(x+1)/self.nbins for x in [*range(self.nbins-1)]]
        partitions = second_difs.quantile(quantiles)['Price'].tolist()
        return partitions

    def get_val(self, data: pd.DataFrame):
        """Get the metric's latest value on the data given."""
        # get the most recent first difference in the left-most column
        dif1 = data.tail(3).diff()[data.columns.values[0]]
        dif2 = dif1.diff().iloc[-1]

        # return the number of partitions to the left of the value
        val = 0
        while dif2 >= self.partitions[val] and val < len(self.partitions)-1: 
            val += 1
        if dif2 >= self.partitions[len(self.partitions)-1]: # case where dif is greater than all partition values
            val += 1 

        return val

def dif1_test():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # create metric
    markov_memory = 1
    bins = 10
    dif1 = Dif1(markov_memory, bins, training_data)

    i = dif1.get_val(testing_data.iloc[-100:-99])
    print(i)

def dif2_test():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # create metric
    markov_memory = 1
    dif2 = Dif2(markov_memory,10,training_data)

    sample_point = -125
    i = dif2.get_val(testing_data.iloc[sample_point-3:sample_point])
    print(i)

if __name__ == '__main__':
    dif2_test()
    