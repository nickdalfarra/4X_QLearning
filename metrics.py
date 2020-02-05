"""Metric objects."""

class Metric():
    """Metric usable by the RL agent."""
    def __init__(self, name: str, memory:int = 1, bins:int = None):
        self.id = metric_id_dict[name] # getting id ensures existence
        self.name = name
        self.memory = memory
        self.bins = bins
        self.num_vals = bins**memory

# global dictionary for mapping metric names to their ids
metric_id_dict = {
    "dif1":0,
    "dif2":1
}

