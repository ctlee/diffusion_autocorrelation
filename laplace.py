#!/bin/env python2.7
"""
@description: 
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: August 25, 2014
HISTORY:
"""
import argparse, logging, math, multiprocessing, os, sys, pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


t = np.arange(0, 10000, 1)
f = t
plt.plot(t,f)
plt.show()

# F(s) = \int_0^{\infty} e^{-st}f(t)dt
s = np.arange(-1, 1, 0.001)
s = s.reshape((len(s),1))
M = np.exp(-s*t)            # unitless

LT = np.dot(M,f*0.01)   # The value of the LT m^2/s
print LT

s = np.arange(-1, 1, 0.001)
plt.plot(s,LT)
plt.show()
