"""Generate codebooks and partitions for metric definitions."""

from pandas import DataFrame, read_csv
from math import ceil

def dist(x: float, y: float):
    """Distance function used in optimization."""
    return (x-y)**2

def expected_error(data: DataFrame, partitions: list, codebook: list):
    """Average distance between data and their quantized values."""
    
    p = 0 # index of quantizer output
    total = 0 # sum of errors
    for i in range(len(data)):
        if p == len(partitions):
            break
        elif data.iloc[i][0] < partitions[p]: 
            pass
        else:
            p += 1

        total += dist(data.iloc[i][0], codebook[p])

    for j in range(i, len(data)): # last bin
        total += dist(data.iloc[j][0], codebook[p])

    return total/len(data)


def nearest_neighbour_condition(data: DataFrame, codebook: list):
    """Produce partitions for regions satisfying the nnc."""

    p = 0
    partitions = []
    for i in range(len(data)):
        if p == len(codebook)-1:
            break
        elif abs(data.iloc[i][0] - codebook[p]) < abs(data.iloc[i][0] - codebook[p+1]):
            pass
        else: # we've passed where a partition should be
            partitions.append((data.iloc[i-1][0]+data.iloc[i][0])/2) # set partition between the two points
            p += 1
    
    return partitions


def centroid_condition(data: DataFrame, partitions: list):
    """Produce codebook satisfying the MSE centroid condition."""

    codebook = []

    p = 0
    R = [] 
    for i in range(len(data)):
        if p == len(partitions):
            break
        elif data.iloc[i][0] < partitions[p]:
            R.append(data.iloc[i][0])
        else:
            codebook.append(sum(R)/len(R))
            R = [data.iloc[i][0]]
            p += 1

    #last bin
    total = 0
    for j in range(i,len(data)):
        total += data.iloc[j][0]

    codebook.append(total/(len(data)-i))

    return codebook


def lloyd_max(data: DataFrame, N: int, epsilon: float):
    """Perform Lloyd-Max iterations to approach an optimal N-level quantizer."""

    data = data.sort_values(by=['Price']) 
    
    # first guess for partitions is the quantiles
    quantiles = [(x+1)/N for x in [*range(N-1)]]
    partitions = data.quantile(quantiles)[data.columns.values[0]].tolist()
    codebook = centroid_condition(data, partitions) # build codebook based on partitions guess
    err1 = expected_error(data, partitions, codebook)
    # print("Error at initialization: " + str(err1))

    # first lloyd max iteration
    partitions = nearest_neighbour_condition(data, codebook)
    codebook = centroid_condition(data, partitions)
    err2 = expected_error(data, partitions, codebook)
    # iterations = 1
    # print("Error after iter "+str(iterations)+": " + str(err2))

    while (err1-err2)/err1 > epsilon:
        partitions = nearest_neighbour_condition(data, codebook)
        codebook = centroid_condition(data, partitions)
        # iterations += 1
        err1 = err2
        err2 = expected_error(data, partitions, codebook)
        # print("Error after iter "+str(iterations)+": " + str(err2)+", % change: "+str(100*(err1-err2)/err1))


    return partitions, codebook
    

def main():
    # prep data
    prices = read_csv('USD_CADHistoricalData.csv', header = 0, usecols = ['Price'])
    # training_split = 0.8
    # training_data = prices.head(ceil(len(prices)*training_split))
    # testing_data = prices.tail(floor(len(prices)*(1-training_split)))
    # del prices

    print(lloyd_max(prices, 10, 0.0001))

if __name__ == "__main__":
    main()