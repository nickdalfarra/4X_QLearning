#matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import io, base64, os, json, re
import pandas as pd
import itertools
import numpy as np
import datetime
from random import randint
import string

#at the bottom i show how to call the markovchain class
#uncomment all lines outside the class below the class 

class MarkovChain:
    
    def __init__(self, memory, binsnums):
        binsnums = binsnums
        bins_labels = MarkovChain.bins_label_list(binsnums)
        print(bins_labels)
        memory = memory
        USDCAD_df = pd.read_csv('USD_CAD.csv')
        USDCAD_df['Date'] = pd.to_datetime(USDCAD_df['Date'])
        new_set = [] 
        market_subset = USDCAD_df.iloc[1:len(USDCAD_df)]
        Date = market_subset['Date']    
        Price_Gap = (market_subset['Price'] - market_subset['Price'].shift(1)) / market_subset['Price'].shift(1)
        Open_Gap = market_subset['Open'].pct_change()
        High_Gap = market_subset['High'].pct_change()
        Low_Gap = market_subset['Low'].pct_change() 
        Daily_HiLow = (market_subset['High'] - market_subset['Low'])# / market_subset['Price']
        Outcome_Next_Day_Direction = (market_subset['Price'].shift(-1) - market_subset['Price'])
        print(len(market_subset))
        new_set.append(pd.DataFrame({'Date': Date,
                                    'Price_Gap':Price_Gap,
                                    'High_Gap':High_Gap,
                                    'Low_Gap':Low_Gap,
                                    'Daily_HiLow':Daily_HiLow,
                                    'Outcome_Next_Day_Direction':Outcome_Next_Day_Direction
                                    }))
        new_set_df = pd.concat(new_set)
        new_set_df = new_set_df.dropna(how='any') 
        new_set_df['Price_Gap_LMH'] = pd.qcut(new_set_df['Price_Gap'], binsnums, labels = bins_labels)
        # High_Gap - not used in this example
        #new_set_df['High_Gap_LMH'] = pd.qcut(new_set_df['High_Gap'], 3, labels=["L", "M", "H"])
        new_set_df['Next_Day_Outcome'] = new_set_df['Outcome_Next_Day_Direction']
        # new set
        new_set_df = new_set_df[["Date", 
                         "Price_Gap_LMH", 
                         #"High_Gap_LMH", 
                         "Next_Day_Outcome"]]
        new_set_df['Event_Pattern'] = new_set_df['Price_Gap_LMH'].astype(str) #+ new_set_df['High_Gap_LMH'].astype(str) + new_set_df['Low_Gap_LMH'].astype(str) + new_set_df['Daily_HiLow_LMH'].astype(str)
        newstuff = []
        newstuff = list(itertools.product(bins_labels, repeat = memory))
        header = []
        nums_array = []
        m = len(new_set_df)
        for i in range(len(newstuff)):
            count = 0
            #print(''.join(newstuff[i]))
            header.append(''.join(newstuff[i]))
            for j in range(2, m-1-memory):
                current_count = 1
                for k in range(len(newstuff[i])):
                    if new_set_df.loc[j + k, 'Event_Pattern'] != newstuff[i][k]:
                        current_count = 0
                count += current_count
            nums_array.append(count)  
        total_count = pd.DataFrame(nums_array, columns = ['count'], index = header)
        total_count.T
        self.df = pd.DataFrame( index = bins_labels, columns = header)
        m = len(new_set_df)
        for i in range(len(newstuff)):
            for a in bins_labels: 
                count = 0
                for j in range(2, m-1-memory):
                    current_count = 0
                    if new_set_df.loc[j + memory, 'Event_Pattern'] == a:
                        current_count = 1
                        for k in range(memory):
                            if new_set_df.loc[j + k, 'Event_Pattern'] != newstuff[i][k]:
                                current_count = 0
                    count += current_count
                if total_count.loc[''.join(newstuff[i]), 'count'] != 0: self.df.loc[a,''.join(newstuff[i])] = count / total_count.loc[''.join(newstuff[i]), 'count']
        print(self.df.T)
        
    def bins_label_list(binsnums):
        bins_labels = []
        for i in range(binsnums):
            bins_labels.append(string.ascii_letters[i])
        return bins_labels


    def table(self):
        table = self.df.T
        return table

    

               
mem = 4
binsnums = 4

s = MarkovChain(mem, binsnums)
table = s.table()
print(table)
#row = table.loc["aaaa" : "aaaa"]
#topid = row.astype(float).idxmax(axis = 1)
#top =  row.max(axis = 1 )
#print(topid)
#print(top)
#print(row)
#print("done")