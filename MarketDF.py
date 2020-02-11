import matplotlib
import matplotlib.pyplot as plt
import io, base64, os, json, re
import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime
from random import randint
import matplotlib.pyplot as plt

#to initialize call everything as a method
#s = Market()
#price = s.Price()
#gap = s.Gap()
#Open = s.Open()
#and so on
#after first call dont use brackets
#now:  
#price = s.Price
#gap - s.Gap
#to move to next day
#s.__next__()
#to move to previous day
#s.__previous__()

class Market:
    def __init__(self):
        USDCAD_df = pd.read_csv('USD_CAD.csv')
        USDCAD_df['Date'] = pd.to_datetime(USDCAD_df['Date'])
        Gap = USDCAD_df['Price'].pct_change()
        Price = USDCAD_df['Price']
        Open = USDCAD_df['Open']
        High = USDCAD_df['High']
        Low = USDCAD_df['Low']
        Change = USDCAD_df['Change %']

        Data = {'Pct_Change': Gap, 
            'Price':Price,
            'Date': USDCAD_df['Date'],
            'Open': USDCAD_df['Open'],
            'High': USDCAD_df['High'],
            'Low': USDCAD_df['Low'],
            'Change': USDCAD_df['Change %']
            }
  
        self.df = DataFrame(Data,columns=['Pct_Change','Price','Date', 'Open', 'High', 'Low', 'Change'])
        self.df.sort_values(by=['Date'], ascending = 'True')
        self.index = 0 



    def Price(self):
        price = self.df.loc[self.index,'Price']
        return price

    def Gap(self):
        gap = self.df.loc[self.index,'Pct_Change']
        return gap

    def Date(self):
        date = self.df.loc[self.index,'Date']
        return date

    def Open(self):
        open = self.df.loc[self.index,'Open']
        return open

    def High(self):
        high = self.df.loc[self.index,'High']
        return high

    def Low(self):
        low = self.df.loc[self.index,'Low']
        return low

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        index = self.index
        self.Previous = self.Price #old rate from yesterday 
        self.Price = Market.Price(self)
        self.Date = Market.Date(self)
        self.Gap = Market.Gap(self)
        self.Open = Market.Open(self)
        self.High = Market.High(self)
        self.Low = Market.Low(self)

    def __previous__(self):
        self.index -= 1 
        index = self.index
        self.Next = self.Price #old rate from yesterday 
        self.Price = Market.Price(self)
        self.Date = Market.Date(self)
        self.Gap = Market.Gap(self)
        self.Open = Market.Open(self)
        self.High = Market.High(self)
        self.Low = Market.Low(self)



#print("price:")
#print(s.Price
#print(s.Gap())
#s.__next__()
#print(s.Price)
#print(s.Gap)
#s.__next__()
#print(s.Gap)
#print(s.gap)


#dates = []
#values = []
#difference = []
#for x in range(1000):
#    dates.append(s.Date)
#    values.append(s.Price)
#    difference.append(s.Gap)
#    s.__next__()

#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
#ax1.plot(dates, values)
#ax2.plot(dates, difference)
#ax3.hist(values, bins = 20)
#ax4.hist(difference, bins = 20)

#plt.show()