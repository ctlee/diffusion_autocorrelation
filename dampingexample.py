#!/bin/env python2.7
"""
@description: Plot the typical damping differential equation solutions
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: November 24, 2014
HISTORY:
    March 1, 2014 -  Made nice plots
SAMPLE RUN: Codeine like molecule
    ./model.py -k 0.1 -m 299 -r 10 -t -p
"""
from __future__ import division
import argparse, math, random, pdb
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


matplotlib.rc('font', **{'size':14})
label_font = {'size':18}

# Defining initial conditions
pos = 2 
vel = 0
force = 0
length = 15         # seconds -8
dt = 0.1             # seconds 
T = 0                             # Kelvin
chunksize = int(length/dt/100)     # chunksize to average over for the plot
bias = 0 

if __name__ == '__main__':
    # Some default values
    k = 1 # Spring constant
    cs = [0, 1, 2, 3] # Damping constant
    m = 1   # Mass
    """
    FIGURE 1: Forces, Noises, Position, Velocities
    """
    fig1 = plt.figure(1, facecolor='white', figsize=(10,8))
    ax1 = fig1.add_subplot(111)

    for c in cs:
        
        pos = 2 
        vel = 0
        force = 0
        print 'Damping Ratio = ' + str(c/(2*math.sqrt(m*k)))    # unitless
        xi = c/(2*math.sqrt(m*k))
        noise = 0 
        b = 1/(1+c*dt/(2*m))    # unitless
        a = (1 - c*dt/(2*m))/(1 + c*dt/(2*m))   # unitless
    
        N = int(math.ceil(length/dt))
        random.seed(12593021)   # my favorite number
        
        # Preallocate the arrays for the results 
        positions = np.zeros(N)
        noises = np.zeros(N)
        forces = np.zeros(N)
        KE = np.zeros(N)
        PE = np.zeros(N)
        energies = np.zeros(N)
        velocities = np.zeros(N)

        for i in xrange(0,N):   # interate steps
            positions[i] = pos
            bump = random.gauss(0,noise)    # Generate the random kick with variance noise
            noises[i] = bump
            """
            This integrator is an implementation of:
            Gronbech-Jensen, N., Farago, O., Molecular Physics, V111, 8:989-991, (2013)
            """
            pos = pos + b*dt*vel + b*dt*dt*force/(2*m) + b*dt/(2*m)*bump    # A
            fnew = -k*pos + bias    # harmonic oscillator + underlying PMF
            forces[i] = fnew        # kg*A*s^-2
            vel = a*vel + dt/(2*m)*(a*force + fnew) + b/m*bump # A/s
            force = fnew
            velocities[i] = vel
            KE[i] = .5*m*vel**2
            PE[i] = .5*k*pos**2
            energies[i] = .5*m*vel**2 + .5*k*pos**2 # kgA^2/s^2

        print "===== Done generating sample trajectory! ====="

        t = np.linspace(0, length, num=N, endpoint=False)
        # Plot the position 
        ax1.plot(t, positions, '-', label=r'$\xi = %0.1f$'%(xi))
    ax1.margins(0.025,0.05)
    ax1.set_ylabel(r'Position [au]', **label_font)
    ax1.set_xlabel(r'Time [au]', **label_font)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, 'lower right')

    plt.tight_layout(pad=0.25)
    plt.show()
