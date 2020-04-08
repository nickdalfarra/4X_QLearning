#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 14:31:47 2020

@author: matthewbourque
"""

import matplotlib.pyplot as plt

class Market:
  def __init__(self):
    # long = 1, short = -1
    self.bias = 0.15
    self.trade_cnt = 0  
     
    self.curr_pos = 0
    self.next_pos = 0

    self.bought_time = []
    self.bought_price = []

    self.sold_time = []
    self.sold_price = []

  def update_pos(self, next_action, prev_action, price, price_idx):
    if next_action > prev_action + self.bias:
      self.curr_pos = 1
      self.bought_time.append(price_idx)
      self.bought_price.append(price)
      self.trade_cnt +=1
    
    elif next_action < prev_action - self.bias:
      self.curr_pos = - 1
      self.sold_time.append(price_idx)
      self.sold_price.append(price)
      self.trade_cnt +=1

    else:
      self.curr_pos = 0
      
    return self.curr_pos




