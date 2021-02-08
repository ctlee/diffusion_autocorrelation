#!/bin/env python2.7

import sys, os, subprocess 
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

matplotlib.rc('font', **{'size':14})
label_font = {'size':18}

files = [0, 0.1, 1, 5, 10, 25, 100]
#files = [0, 0.01, 0.1, 1, 5, 10, 25, 100]
#files = [0.1, 1, 5]
#files = [1]

fig1 = plt.figure(1, facecolor='white', figsize=(10,8))
ax1 = fig1.add_subplot(111)
#ax1.set_xlabel(r'Time [$ns$]')
ax1.set_xlabel(r'Colvars Period')
#ax1.set_ylabel(r'$\langle \delta r(0) \delta r(t)\rangle$ [$A^2$]')
ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^6$]')
#ax1.set_ylabel(r'Percent of X(0)')

for file in files:
    #data = np.loadtxt('acr_step25_k%s'%(file))
    data =  np.loadtxt('datasets/long_m299_r10_k%s'%(file))
    #data = np.loadtxt('var_tau_m299_r10_k%s'%(file))
    if file == 0:
        #ax1.plot(data[:,0], data[:,2]*10**6, label=r'vacr')
        ax1.plot(data[:,0], data[:,3]*10**6, label=r'Einstein', linewidth=3)
    ax1.plot(data[:,0], data[:,1]*10**6, label=r'K=%s'%(file))
    #ax1.plot(data[:,0], data[:,2]*10**6, label=r'greenKubo')
    #ax1.plot(data[:,0], data[:,1], label=r'K=%s'%(file))
    #ax1.plot(data[:,0], data[:,1]/data[0,1], label=r'k=%s tau'%(file))
    #ax1.plot(data[:,0], data[:,2]/data[0,2], label=r'k=%s var'%(file))
handles, labels  = ax1.get_legend_handles_labels()
ax1.legend(handles, labels)
plt.tight_layout(pad=0.5)
plt.show()
