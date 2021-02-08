#!/bin/env python2.7
"""
@description: Iterate over the namd trajectories and compare with the model
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: February 27, 2015
HISTORY:
"""

import argparse, math, os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import model, plottools

matplotlib.rc('font', **{'size':14})
label_font = {'size':18}

if __name__ == '__main__':
    chunksize = 10
    zmeansim = np.zeros(71)
    zvarmeansim = np.zeros(71)
    
    fig1, (ax1, ax2) = plt.subplots(1,2, sharey=True, 
            facecolor = 'white', figsize = (10,8))
    length = 2*math.pow(10,-8)
    dt = 2*math.pow(10,-15)
    t = np.arange(0,length, dt*2500) * math.pow(10,9)
    ax1.set_xlabel(r'Time [$ns$]', **label_font) 
    ax1.set_ylabel(r'Position [$\AA$]', **label_font)
    ax1.set_title('Simulated Trajectories')
    ax1.set_ylim([-3,3])
    ax1.margins(0.025, 0.05) 
    for i in xrange(0,71,1):
        traj = model.genTrajC(k = 1.5,
            mass = 299,
            r = 7.3,
            length = length,
            dt = dt,
            viscosity = 3.21e-14,
            random = 1)
        z = traj[0][::2500]
        max_env, min_env, xcenters, ycenters = plottools.chunk(t, z, chunksize)
        ax1.fill_between(xcenters, min_env, max_env, color="#70acfb",
                alpha=0.25)
        zmeansim[i] = np.mean(z)
        zvarmeansim[i] = np.var(z)
        #print '<z> = ' + str(zmeansim[i]) + '; var(z) = ' + str(zvarmeansim[i])
    ax1.plot(t,z, 'k-', linewidth=0.5)
    ax1.text(0, -2.75, r'$\langle z\rangle = %0.5f$; $\langle\sigma^2\rangle = %0.5f$'%(np.mean(zmeansim), np.mean(zvarmeansim)), fontsize=18)
    print 'GLOBAL STATS FROM MODEL'
    print '<<z>> = ' + str(np.mean(zmeansim)) + '; <var(z)> = ' + str(np.mean(zvarmeansim))

    windows = xrange(-35, 36, 1)
    zmean = np.zeros(71)
    zvarmean = np.zeros(71)
    ax2.set_xlabel(r'Time [$ns$]', **label_font) 
    ax2.set_title(r'Trajectories From MD Experiment')
    ax2.margins(0.025, 0.05) 
    ax2.set_color_cycle(['r', 'g', 'b'])

    for window in windows:
        #print 'Running window: ' + str(window)
        filename = '/net/home/clee/permeability/simulations/original/Production/cod/window_%d/umb-total.colvars.traj'%(window)
        t = np.loadtxt(filename, comments='#', usecols=(0,), skiprows=1)
        z = np.loadtxt(filename, comments='#', usecols=(1,), skiprows=1)
        z = z - window
        t = t*2*math.pow(10, -6)
        max_env, min_env, xcenters, ycenters = plottools.chunk(t, z, chunksize)
        ax2.fill_between(xcenters, min_env, max_env, color="#70acfb",
                alpha=0.25)
        """
        ax2.plot(xcenters, ycenters, '-', linewidth=0.25, 
                alpha=0.1, label=window)
        """
        zmean[window+35] = np.mean(z)
        zvarmean[window+35] = np.var(z)
        #print '<z> = ' + str(np.mean(z)) + '; var(z) = ' + str(np.var(z))
    ax2.plot(t,z, 'k-', linewidth=0.5)
    ax2.text(0, -2.75, r'$\langle z\rangle = %0.5f$; $\langle\sigma^2\rangle = %0.5f$'%(np.mean(zmean), np.mean(zvarmean)), fontsize=18)

    
    print 'GLOBAL STATS FROM EXPERIMENT'
    print '<<z>> = ' + str(np.mean(zmean)) + '; <var(z)> = ' + str(np.mean(zvarmean))
    plt.tight_layout(pad=0.5)
    plt.savefig('./Figures/colvarsCompare.png', dpi = 300)
    plt.savefig('./Figures/colvarsCompare.eps', dpi = 1200)
    plt.show()
