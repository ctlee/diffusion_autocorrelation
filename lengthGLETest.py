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

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(process)d %(processName)s: %(message)s',)

fn = 'lengthGLETest.txt'

"""
A worker process to calculate the D(z) for a particular k.
Prototype:
    arg[0]: (double) k - spring constant
    arg[1]: (mp.queue) q - queue to watch
    arg[2]: (double) length - length to run
    arg[3]: (int) randseed - np.seed
"""
def worker(args):
    k = args[0]
    out_q = args[1]
    length = args[2]*math.pow(10,-8)
    np.random.seed(args[3])
    dt = 2*math.pow(10,-15)
    traj = model.genTrajC(k,
            mass = 299,
            r = 7.3,
            length=length,
            dt = dt,
            verbosity = 0,
            random = 1)
    slice = 1
    vacr = model.fftAutocorrelation(traj[1][::slice])
    
    # Calculate the laplace transform of vacr
    N = len(vacr)
    z = np.arange(0.1, 2, 0.1)
    M = z.reshape((len(z),1))*np.ones(N)
    M = np.power(M,-np.arange(N))
    LT = np.dot(M,vacr)             # The value of the LT

    pvar = np.var(traj[0])          # Variance of the position
    mvel = np.mean(traj[1]**2)      # Mean of the velocities squared
    pdb.set_trace()
    Ds = LT*pvar*mvel/(LT*(z*pvar - mvel/z)-pvar*mvel)
    plt.plot(z,Ds)
    plt.show() 

    out_q.put("%0.2f %0.4e"%(args[2], D))

"""
A listener to watch the queue and write to a file:
Prototype: 
    arg[0]: (str) fname - filename
    arg[1]: (mp.queue) q - queue to watch
"""
def writer(q):
    fd = open('datasets/%s'%(fn), 'wb+')
    while 1:
        m = q.get()
        if m =='kill':
            f.write('killed')
            break
        fd.write(str(m) + '\n')
        fd.flush()
    fd.close()

def run():
    manager = multiprocessing.Manager()
    out_q = manager.Queue()
    pool = multiprocessing.Pool(maxtasksperchild=1, processes=2)
    watcher = pool.apply_async(writer, (out_q, ))

    lengths = np.arange(0.25, 10, 0.25)
    jobs = []
    for length in lengths:
        for i in np.arange(0, 100, 1):
            jobs.append((0, out_q, length, i))
    pool.map(worker, iter(jobs))
    out_q.put('kill')   # Kill the writer
    pool.close()
    pool.join()

def plot():
    data = np.loadtxt('datasets/%s'%(fn))
    d = defaultdict(np.array)
    stacked = np.array([])
    for row in data:
        row[1] = row[1]*10**6
        if d.has_key(row[0]*10):
            d[row[0]*10] = np.append(d[row[0]*10], row[1])
        else:
            d[row[0]*10] = np.array(row[1])
    keylist = d.keys()
    keylist.sort()
    for key in keylist:
        if stacked.size == 0:
            stacked = d[key].T
        else:
            stacked = np.column_stack((stacked, d[key].T))
    
    matplotlib.rc('font', **{'size':14})
    label_font = {'size':18}
    fig1 = plt.figure(1, facecolor='white', figsize=(10,8))
    
    ax1 = fig1.add_subplot(111)
    ax1.margins(0.025, 0.05)
    max_env = stacked.max(axis=0)
    min_env = stacked.min(axis=0)
    mean =  stacked.mean(axis=0) 
    ax1.fill_between(keylist, min_env, max_env, color='gray', edgecolor='none', alpha=0.3)
    ax1.plot(keylist, mean, '-', linewidth=2, linestyle='-', color='k', label='Mean') 
    ## CI 0.99
    mean, low_env, high_env = plottools.confidenceInterval(stacked, .99)
    ax1.fill_between(keylist, low_env, high_env, color='red', edgecolor='none', alpha = 0.6, label='CI 0.99')
    ## CI 0.95
    mean, low_env, high_env = plottools.confidenceInterval(stacked, .95)
    ax1.fill_between(keylist, low_env, high_env, color='green', edgecolor='none', alpha = 0.6, label='CI 0.95')
    ## CI 0.5
    mean, low_env, high_env = plottools.confidenceInterval(stacked, .5)
    ax1.fill_between(keylist, low_env, high_env, color='blue', edgecolor='none', alpha = 0.6, label='CI 0.5')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_ylim([3.25, 3.7])
    ax1.set_xlabel(r'Simulation Length [$ns$]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^6$]')
    ### HACK ###
    red_patch = mpatches.Patch(color='red', alpha=0.6, label='CI 0.99')
    green_patch = mpatches.Patch(color='green', alpha=0.6, label='CI 0.95')
    blue_patch = mpatches.Patch(color='blue', alpha=0.6, label='CI 0.5')
    newleg = plt.legend(handles=[red_patch, green_patch, blue_patch], loc=1)
    ax1.add_artist(newleg)
    ############
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc=4)

    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/lengthGLE.png', dpi=300, transparent=True)
    plt.show()

# BEGIN MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compare the effect of simulation length on the convergence of the runs")
    parser.add_argument('-r',
            action = 'store_true',
            dest = 'run',
            help = 'Run the analysis')
    parser.add_argument('-p',
            action = 'store_true',
            dest = 'plot',
            help = 'Plot the results')
    parser.add_argument('-f',
            action = 'store',
            dest = 'fn',
            help = 'filename to store output/input')
    args = parser.parse_args() 
    if len(sys.argv) < 2:
        args = parser.parse_args(['-h'])
        exit()
    if args.fn:
        fn = str(args.fn)
    if args.run:
        run()
    if args.plot:
        plot()
