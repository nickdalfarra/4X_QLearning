"""Metrics usable by the worker. Metrics must be given training data at initialization."""

import pandas as pd
from math import floor, ceil

class Metric():
    """Base metric class."""
    def __init__(self, pts_required: int, metric_method: object, nbins: int, markov_mem: int):
        self.pts_required = pts_required
        self.metric_method = metric_method
        self.nbins = nbins
        self.markov_mem = markov_mem  # Markov memory will influence stochastic kernel fitting, not here
        self.n_vals = nbins**(markov_mem+1)
        self.partitions = None
    
    def fit(self, training_data: pd.DataFrame):
        """Creates the partitions which define the metric mapping."""
        historical_values = self.metric_method(training_data)
        quantiles = [(x+1)/self.nbins for x in [*range(self.nbins-1)]]
        self.partitions = historical_values.quantile(quantiles)['Price'].tolist()

    # TODO: incorporate Markov memory here in get_val!!!!!!!!!!

    def get_val(self, data: pd.DataFrame):
        """Get the metric's latest value on given data."""

        if self.partitions is None:
            raise AttributeError("Metric must be fit to training data using the .fit() method.")

        # get the most recent metric value in the data's first column (input should only be one column)
        dif = self.metric_method(data.tail(self.pts_required))[data.columns.values[0]].iloc[-1]

        # return the number of partitions to the left of the value
        val = 0
        while dif >= self.partitions[val] and val < len(self.partitions)-1: 
            val += 1
        if dif >= self.partitions[len(self.partitions)-1]: # case where dif is greater than all partition values
            val += 1 

        return val


class Dif1(Metric):
    """Bins on the first difference distribution."""
    def __init__(self, markov_mem: int, nbins: int):
        pts_required = 2 # points required for the metric to be calculated
        super().__init__(pts_required, self.first_dif, nbins, markov_mem)
        
    def first_dif(self, training_data: pd.DataFrame):
        """How the metric calculates values."""
        return training_data.diff()

class Dif2(Metric):
    """Bins on the second difference distribution."""
    def __init__(self, markov_mem: int, nbins: int):
        pts_required = 3 # points required for the metric to be calculated
        super().__init__(pts_required, self.second_dif, nbins, markov_mem)

    def second_dif(self, training_data: pd.DataFrame):
        """How the metric calculates values"""
        return training_data.diff().diff()


def dif1_test():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # create metric
    markov_memory = 1
    bins = 10
    dif1 = Dif1(markov_memory, bins)
    dif1.fit(training_data)

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
    nbins = 10
    dif2 = Dif2(markov_memory, nbins)
    dif2.fit(training_data)

    sample_point = -125
    i = dif2.get_val(testing_data.iloc[sample_point-3:sample_point])
    print(i)

if __name__ == '__main__':
    dif1_test()
    