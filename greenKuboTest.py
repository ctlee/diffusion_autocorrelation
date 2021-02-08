#!/bin/env python2.7
"""
@description: 
@author: Christopher T. Lea (ctlee@ucsd.edu)
LAST TOUCHED: August 25, 2014
HISTORY:
"""
import argparse, logging, math, multiprocessing, os, subprocess, sys, threading, time
import numpy as np
import model
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(process)d %(processName)s: %(message)s',)

"""
A worker process to calculate the D(z) for a particular k.
Prototype:
    arg[0]: (double) k - spring constant
    arg[1]: (mp.queue) q - queue to watch
"""
def worker(args):
    k = args[0]
    out_q = args[1]
    length = 2*math.pow(10,-8)
    dt = 2*math.pow(10,-15)
    slice = 1 
    positions, forces, velocities, noises, KE, PE, energies = model.generateTrajectoryC(k,
            mass = 299,
            r = 10,
            length=length,
            dt = dt,
            verbosity = 0)
    del positions, forces, noises, KE, PE, energies
    
    vacr = model.fftAutocovariance(velocities[0::slice])
    vacr = vacr[0:int(len(vacr)/2)]             # Truncate by half   
    D = np.sum(vacr)*dt*slice*math.pow(10,-16)  # cm^2/s
    outdict={k:D}
    out_q.put(outdict)

"""
A listener to watch the queue and write to a file:
Prototype: 
    arg[0]: (str) fname - filename
    arg[1]: (mp.queue) q - queue to watch
"""
def writer(args):
    fd = open(args[0], 'wb+')
    while 1:
        m = args[1].get()
        if m =='kill':
            break
        fd.write(str(m) + '\n')
        fd.flush()
    fd.close()


# BEGIN MAIN
if __name__ == '__main__':
    processes = []
    manager = multiprocessing.Manager()
    out_q = manager.Queue()
    pool = multiprocessing.Pool( maxtasksperchild=1)

    watcher = pool.apply_async(writer, ('testfile', out_q,))

    #ks = np.logspace(0, 2, num=50)-1
    ks = np.arange(0, 100, 1) 
    jobs = list()
    for k in ks:
        jobs.append((k, out_q))
    pool.map(worker, iter(jobs))
    pool.close()
    pool.join()

    resultdict = {}
    while not out_q.empty():
        resultdict.update(out_q.get())
    
    spring = list()
    D = list()
    for k,v in resultdict.iteritems():
        spring.append(k)
        D.append(v*10**6)
    spring, D = (list(x) for x in zip(*sorted(zip(spring, D))))

    # Matplotlib rc
    matplotlib.rc('font', **{'size':14})
    label_font = {'size':18}

    fig1 = plt.figure(1, facecolor='white', figsize=(10,8))
    ax1 = fig1.add_subplot(111)
    ax1.plot(spring, D, '-')
    ax1.axhline(y=2.44152312438)
    ax1.set_xlabel(r'Spring Constant [$kcal/mol/A^2$]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^6$]')
    #ax1.set_xscale('log')
    plt.show()
