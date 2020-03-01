#matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
#import io, base64, os, json, re
import pandas as pd
import itertools
import numpy as np
import datetime
from random import randint


class MarkovChain:
    
    def __init__(self, mem):
        #binsnums = binsnums
        binsnums = 10
        bins_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        self.mem = mem
        #self.memory = 1
        #next = MarkovChain.hi(memory)#self.t = 1


    def hi(self):
        what = self.mem + 1 
        return what
               
mem = 1
binsnums = 10
bins_labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


s = MarkovChain(mem)
table = s.hi()
print(table)
print("done")