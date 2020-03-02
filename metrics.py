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

    def val(self, data: pd.DataFrame):
        """Get the metric's latest value on given data."""

        if self.partitions is None:
            raise AttributeError("Metric must be fit to training data using the .fit() method.")

        val_sequence = []
        for i in range(self.markov_mem + 1): 
            data_slice = data.iloc[len(data)- i - self.pts_required : len(data) - i]
            metric_val = self.metric_method(data_slice)[data.columns.values[0]].iloc[-1]
            
            # find appropriate bin
            bin_num = 0
            while metric_val >= self.partitions[bin_num] and bin_num < len(self.partitions)-1: 
                bin_num += 1
            if metric_val >= self.partitions[len(self.partitions)-1]: # case where dif is greater than all partition values
                bin_num += 1 

            val_sequence.append(bin_num)
        
        return self.markov_dict[tuple(val_sequence)]


class Price(Metric):
    """Bins on the price distribution."""
    def __init__(self, nbins: int, markov_mem: int):
        pts_required = 1 # points required for the metric to be calculated
        super().__init__(pts_required, self.price, nbins, markov_mem)

    def price(self, training_data: pd.DataFrame):
        """How the metric calculates values."""
        return training_data


class Dif1(Metric):
    """Bins on the first difference of the price distribution."""
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
        """How the metric calculates values."""
        return training_data.diff().diff()


def metric_test(met: Metric):
    # prep data
    prices = pd.read_csv('test_data.csv', header = 0, usecols = ['Price'])
    training_split = 0.8
    training_data = prices.head(ceil(len(prices)*training_split))
    # testing_data = prices.tail(floor(len(prices)*(1-training_split)))

    del prices

    # create metric
    met.fit(training_data)

    # print(met.pts_required + met.markov_mem - 1)
    # print(training_data.iloc[0 : met.pts_required + met.markov_mem ])
    print(met.val(training_data.iloc[0 : met.pts_required + met.markov_mem ]))


if __name__ == '__main__':
    metric_test(Price(8, 1))
    