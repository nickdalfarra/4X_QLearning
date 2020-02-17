from numpy import array
import operator as op
from functools import reduce

class ActionSpace():
    """Probibility mass functions with support on num_vehicles+1 points,
    and a minimum allocation increment of 1/precision."""

    def __init__(self, num_vehicles: int, precision: int):
        self.num_vehicles = num_vehicles + 1
        self.precision = precision
        self.actions = generate_action_space(self.num_vehicles, self.precision)
        self.cardinality = n_Choose_k(self.num_vehicles + self.precision - 1, self.num_vehicles - 1)

    def __repr__(self): return self.actions

    def __str__(self): return str(self.actions)

def n_Choose_k(n: int, k: int):
    """Effecient implementation of combinatorics function."""
    k = min(k, n-k)
    numer = reduce(op.mul, range(n, n-k, -1), 1)
    denom = reduce(op.mul, range(1, k+1), 1)
    return int(numer / denom)

def generate_list(m, n):
    '''Generate all sets of length m whose elements sum to n.'''
    if m == 1: # base case
        yield [n,]
    else:
        for val in range(n + 1):
            for perm in generate_list(m - 1, n - val):
                yield [val,] + perm

def generate_action_space(m, n):
    '''Generate the action space of pmfs with support m and minimum increment 1/n.'''
    return array(list(generate_list(m,n)))/n

def main():
    vehicles = 1
    precision = 1000
    A = ActionSpace(vehicles, precision)
    print(A.actions)
    print(A.cardinality)

if __name__ == "__main__":
    main()
    