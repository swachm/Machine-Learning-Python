#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 19:56:48 2017

@author: manmeetswach
"""
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import style
style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([1,2,3,4,5,6], dtype = np.float64)



def best_fit_slope_and_intercept(xs, ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
         (mean(xs)*mean(xs)) - mean(xs*xs))
    b = mean(ys) - m*mean(xs)
    return m,b

m,b = best_fit_slope_and_intercept (xs,ys)
print (m,b)

regression_line = [(m*x)+b for x in xs]

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()