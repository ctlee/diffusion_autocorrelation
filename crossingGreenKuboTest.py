#!/bin/env python2.7
"""
@description: Investigate the effect of cutoff on the Green Kubo estimate
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: September 21, 2015
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

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(process)d %(processName)s: %(message)s',)

fn = 'crossingGreenKubo.txt'

"""
A worker process to calculate the D(z) for a particular k.
Prototype:
    arg[0]: (double) k - spring constant
    arg[1]: (mp.queue) q - queue to watch
    arg[2]: (int) randseed - np.seed
"""
def worker(args):
    k = args[0]
    out_q = args[1]
    length = 10*math.pow(10,-8)
    np.random.seed(args[2])
    dt = 2*math.pow(10,-15)
    traj = model.genTrajC(k,
            mass = 299,
            r = 7.3,
            length=length,
            dt = dt,
            verbosity = 0,
            random = 1)
    slice = 1
    vacr = model.fftAutocovariance(traj[1][::slice])
    index = np.where(np.diff(np.sign(vacr)))[0]
    if len(index):
        for crossing in np.arange(0,10,1):
            vacrtrunc = vacr[0:index[crossing]]
            Dvacr = np.sum(vacrtrunc)*dt*slice*math.pow(10,-16)  # cm^2/s
            out_q.put("%d %0.4e"%(crossing, Dvacr))
    else:
        vacrtrunc = vacr
        Dvacr = np.sum(vacrtrunc)*dt*slice*math.pow(10,-16)  # cm^2/s
        out_q.put("%d %0.4e"%(-1, Dvacr))

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
    pool = multiprocessing.Pool(maxtasksperchild=1, processes=10)
    watcher = pool.apply_async(writer, (out_q, ))
    jobs = []
    for i in np.arange(0, 100, 1):
        jobs.append((0, out_q, i))
    pool.map(worker, iter(jobs))
    out_q.put('kill')   # Kill the writer
    pool.close()
    pool.join()

def plot():
    data = np.loadtxt('datasets/%s'%(fn))
    d = defaultdict(np.array)
    stacked = np.array([])
    for row in data:
        row[1] = row[1]*10**6       # 1e6
        if d.has_key(row[0]):
            d[row[0]] = np.append(d[row[0]], row[1])
        else:
            d[row[0]] = np.array(row[1])
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
    ax1.set_xlabel(r'Crossing')
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
    plt.savefig('Figures/crossingGreen-Kubo.png', dpi=1200, transparency=True)
    plt.savefig('Figures/crossingGreen-Kubo.eps', dpi=1200)
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
