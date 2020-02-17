"""Metrics usable by the worker."""

import pandas as pd
from math import floor, ceil
from itertools import product

class Metric():
    """Base metric class."""
    def __init__(self, pts_required: int, metric_method: object, nbins: int, markov_mem: int):
        """There are a lot of things that describe a metric!"""
        self.pts_required = pts_required
        self.metric_method = metric_method
        self.nbins = nbins
        self.markov_mem = markov_mem
        possible_sequences = [*product(*[range(nbins) for _ in range(markov_mem+1)])]
        self.markov_dict =  {y:x for x,y in dict(enumerate(possible_sequences)).items()}
        self.n_vals = nbins**(markov_mem+1)
        self.partitions = None # make it none until metric has been fit to do data
    
    def fit(self, training_data: pd.DataFrame):
        """Creates the partitions which define the metric mapping."""
        historical_values = self.metric_method(training_data)
        quantiles = [(x+1)/self.nbins for x in [*range(self.nbins-1)]]
        self.partitions = historical_values.quantile(quantiles)[historical_values.columns.values[0]].tolist()

    def get_val(self, data: pd.DataFrame):
        """Get the metric's latest value on given data."""

        if self.partitions is None:
            raise AttributeError("Metric must be fit to training data using the .fit() method.")

        # get a sequence of metric values of length "markov_mem + 1"
        val_sequence = []
        for i in range(self.markov_mem + 1):
            point = -1 - (self.pts_required + self.markov_mem - i)
            metric_val = self.metric_method(data.iloc[point : point + self.pts_required])[data.columns.values[0]].iloc[-1]
            # find the number of partitions to the left of the value
            bin_num = 0
            while metric_val >= self.partitions[bin_num] and bin_num < len(self.partitions)-1: 
                bin_num += 1
            if metric_val >= self.partitions[len(self.partitions)-1]: # case where dif is greater than all partition values
                bin_num += 1 

            val_sequence.append(bin_num)

        # return the metric's value 
        return self.markov_dict[tuple(val_sequence)]

class Dif1(Metric):
    """Bins on the first difference distribution."""
    def __init__(self, nbins: int, markov_mem: int):
        pts_required = 2   # points required for the metric to be calculated
        super().__init__(pts_required, self.first_dif, nbins, markov_mem)
        
    def first_dif(self, training_data: pd.DataFrame):
        """How the metric calculates values."""
        return training_data.diff()

class Dif2(Metric):
    """Bins on the second difference distribution."""
    def __init__(self, nbins: int, markov_mem: int):
        pts_required = 3 # points required for the metric to be calculated
        super().__init__(pts_required, self.second_dif, nbins, markov_mem)

    def second_dif(self, training_data: pd.DataFrame):
        """How the metric calculates values"""
        return training_data.diff().diff()

class Price(Metric):
    """Bins on the second difference distribution."""
    def __init__(self, nbins: int, markov_mem: int):
        pts_required = 1 # points required for the metric to be calculated
        super().__init__(pts_required, self.price, nbins, markov_mem)

    def price(self, training_data: pd.DataFrame):
        """How the metric calculates values"""
        return training_data

def dif1_test():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # create metric
    markov_memory = 2
    bins = 100
    dif1 = Dif1(bins, markov_memory)
    dif1.fit(training_data)

    point = -98
    i = dif1.get_val(testing_data.iloc[point - dif1.pts_required - dif1.markov_mem: point+1])
    # print(dif1.markov_dict)
    print(i)

def dif2_test():
    # prep data
    prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    # create metric
    markov_memory = 0
    nbins = 10
    dif2 = Dif2(markov_memory, nbins)
    dif2.fit(training_data)

    sample_point = -125
    i = dif2.get_val(testing_data.iloc[sample_point-3:sample_point])
    print(i)

# def price_test():
#     # prep data
#     prices = pd.read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
#     training_split = 0.8
#     training_data = prices.head(ceil(len(prices)*training_split))
#     testing_data = prices.tail(floor(len(prices)*(1-training_split)))

if __name__ == '__main__':
    dif1_test()
    