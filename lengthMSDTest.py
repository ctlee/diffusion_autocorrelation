#!/bin/env python2.7
"""
@description: 
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: August 25, 2014
HISTORY:
"""
import argparse, logging, math, multiprocessing, os, sys
import numpy as np
from collections import defaultdict
from timeit import default_timer as timer
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import model, plottools
import scipy.stats as stats

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(process)d %(processName)s: %(message)s',)

fn = 'lengthMSDTest.txt'

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
    stride = int(25000/slice)# 50ps @2fs ts divided by stride  
    skip = int(2500/slice)

    diff = np.zeros((len(traj[0])/stride, stride))
    for i in np.arange(0, len(traj[0]), stride):
        for j in np.arange(0, stride, 1):
            diff[i/stride][j] = traj[0][i+j] - traj[0][i]
    diff = np.power(diff, 2)
    mean = diff.mean(axis=0)
    tsteps = np.arange(0, 50000, slice*2)
    tskip = tsteps[skip:] 
    mskip = mean[skip:]
    slope, intercept, r_value, p_value, std_err = stats.linregress(tskip, mskip)
    Dmsd = slope/2*0.1 # cm^2/s

    logging.debug("%0.2f %0.4e %f %f"%(args[2], Dmsd, r_value**2, intercept))
    out_q.put("%0.2f %0.4e %f %f"%(args[2], Dmsd, r_value**2, intercept))
   
    """
    fig1 = plt.figure(1, facecolor='white', figsize=(10,8))
    ax1 = fig1.add_subplot(111)
    ax1.margins(0.025, 0.05)

    line = slope*tskip+intercept
    
    ax1.plot(tsteps, mean, linewidth=2)
    ax1.plot(tskip, line, linewidth=2, linestyle='--', color='r',  label=r'$y=%fx$'%(slope) + '\n' + r'$D = %1.2f\times 10^{-5}\, cm^2s^{-1};\quad r^2=%0.3f$'%(Dmsd*1e5, r_value**2))

    ax1.set_xlabel(r'Time [$fs$]')
    ax1.set_ylabel(r'MSD')

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc=4)
    plt.tight_layout(pad=0.5)
    plt.show()
    """

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
    pool = multiprocessing.Pool(maxtasksperchild=1, processes=16)
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
    #ax1.set_ylim([2.5, 4])
    ax1.set_xlabel(r'Simulation Length [$ns$]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
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
    plt.savefig('Figures/lengthMSD.png', dpi=1200, transparent=True)
    plt.savefig('Figures/lengthMDS.eps', dpi=1200)
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
    parser.add_argument('-c',
            action = 'store',
            dest = 'crossing',
            help = 'Which zero crossing to truncate at')
    args = parser.parse_args() 
    if len(sys.argv) < 2:
        args = parser.parse_args(['-h'])
        exit()
    if args.fn:
        fn = str(args.fn)
    if args.crossing:
        crossing = int(args.crossing)
    if args.run:
        run()
    if args.plot:
        plot()