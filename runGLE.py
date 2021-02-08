#!/bin/env python2.7
"""
@description: 
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: August 25, 2014
HISTORY:
"""
import argparse, logging, math, multiprocessing, os, sys, pdb
import numpy as np
from collections import defaultdict
from timeit import default_timer as timer
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import model, plottools
import scipy.stats as stats

matplotlib.rc('font', **{'size':14})
label_font = {'size':18}
dt = 2e-15
traj = model.genTrajC(k=10,
            mass = 299,
            r = 7.3,
            length=2e-8,
            dt = dt,
            verbosity = 0,
            #viscosity = 3.21e-14,
            random = 1)
skip = 1
vels = traj[1][::skip]
dt = dt*skip
vacr = model.fftAutocorrelation(vels)  # A^2/s^2

# Calculate the laplace transform of vacr
# F(s) = \int_0^{\infty} e^{-st}f(t)dt
N = len(vacr)
crossing = 1
index = np.where(np.diff(np.sign(vacr)))[0]
print "Truncating vacr at index %d..."%(index[crossing])
# TODO: Logic here prevent index from behaving badly

vacr = vacr[0:index[crossing]]
t = np.arange(0, index[crossing], 1)*dt   # time s

fig1 = plt.figure(1, facecolor='white', figsize=(10,8))

ax1 = fig1.add_subplot(111)
ax1.margins(0.025, 0.05)
ax1.plot(t*1e12, vacr, linewidth=2, linestyle='-', color = 'k')
ax1.set_xlabel(r'Time [$ps$]')
ax1.set_ylabel(r'VACR')
ax1.set_xlim([0,3])

s = np.linspace(1e10, 5e14,5000)
#s = np.logspace(10,14, num=1000)
sT = s.reshape((len(s),1))
M = np.exp(-sT*t)            # unitless

Cs = np.dot(M,vacr*dt)   # The value of the LT A^2/s

#print "Cs..."
#print Cs

pvar = np.var(traj[0])      # Variance of the position A^2
mvel = np.var(vels)      # Variance of the velocities A^2/s^2

print "pvar var(z) = %f"%(pvar)
print "mvel var(v) = %e"%(mvel)

# ignore divide by zero error for the discontinuity
#with np.errstate(divide='ignore'):
Ds = -Cs*pvar*mvel/(Cs*(s*pvar - mvel/s)-pvar)*1e-16    # cm^2/s
#print "Ds..."
#print Ds

skip = 100

sSkip = s[skip:]
DsSkip = Ds[skip:]

slope, intercept, r_value, p_value, std_err = stats.linregress(sSkip, DsSkip)

line = slope*s+intercept
print "D(z) = %f"%(intercept*1e6) # cm^2/s
#print Ds[-1]

fig2 = plt.figure(2, facecolor='white', figsize=(10,8))

ax1 = fig2.add_subplot(111)
ax1.margins(0.025, 0.05)
ax1.plot(s*1e-12,Cs, 'o', linewidth=2, linestyle='-', color = 'k')

#ax1.set_xlim([0,50])
ax1.set_xlabel(r's [$ps^{-1}$]')
ax1.set_ylabel(r'$C(s)$ [$s$]')

fig3 = plt.figure(3, facecolor='white', figsize=(10,8))

ax1 = fig3.add_subplot(111)
ax1.margins(0.025, 0.05)
ax1.plot(s*1e-12,Ds*1e6, 'o', linewidth=2, linestyle='-', color = 'k')
ax1.plot(s*1e-12, line*1e6, linewidth=2, linestyle='--', color='r')
ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')

#ax1.set_xlim([0,2])
ax1.set_ylim([-10, 10])
ax1.set_xlabel(r's [$ps^{-1}$]')
ax1.set_ylabel(r'$D(s)$ [$cm^2/s\times 10^{-6}$]')
plt.show()
