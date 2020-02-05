"""State space for Q-Learning agents."""
from functools import reduce
import metrics as met

class StateSpace():
    """States of the RL agent.
    'metrics' is a list of Metric type objects."""

    def __init__(self, metrics: list):

        if type(metrics) != "list": TypeError("metrics must be a list of Metric objects.")
        if not metrics: raise ValueError("StateSpace metric list must be non-empty.")

        self.metrics = metrics

        if len(metrics) > 1:
            self.cardinality = reduce(lambda a,b : a.num_vals * b.num_vals, self.metrics)
            self.max_memory = reduce(lambda a,b: max(a.memory, b.memory), self.metrics)
        elif len(metrics) == 1:
            self.cardinality = metrics[0].num_vals
            self.max_memory = metrics[0].memory
            
    def __str__(self): return "state space of size " + str(self.cardinality)

def main():
    dif1 = met.Metric("dif1", 1, 4)
    dif2 = met.Metric("dif2", 3, 3)
    metrics = [dif1, dif2]

    X = StateSpace(metrics)
    print(X)

def type_test():
    X = StateSpace(['a'])

if __name__ == "__main__":
    main()
    # type_test()
