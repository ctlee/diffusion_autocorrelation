#!/bin/env python2.7
"""
@description:  Generate a bunch of trajectories and save them WARNING the files are HUGE
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: March 21, 2015
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
    arg[2]: (double) length - length to run
    arg[3]: (int) randseed - np.seed
"""
def worker(args):
    k = args[0]
    length = args[1]
    np.random.seed(args[2])
    dt = 2*math.pow(10,-15)
    output = model.genTrajC(k,
            mass = 299,
            r = 10,
            length=length,
            dt = dt,
            verbosity = 0,
            random = 1)
    # BEWARE this can generate a TON of data! 
    np.savez_compressed('/scratch/ctlee/k%d_m299_r10_rnd%d'%(k, args[2]), output)

def main():
    pool = multiprocessing.Pool(maxtasksperchild=1)
    jobs = []
    for i in np.arange(0, 1, 1):
        jobs.append((0, math.pow(10,-6), i))
    pool.map(worker, iter(jobs))
    pool.close()
    pool.join()

# BEGIN MAIN
if __name__ == '__main__':
    main()
