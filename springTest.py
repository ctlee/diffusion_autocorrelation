#!/bin/env python2.7
"""
@description: Comparing the effect of spring constant and slice on estimate accuracy
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: March 22, 2015
HISTORY:
"""
import argparse, logging, math, multiprocessing, os, sys
import numpy as np
from collections import defaultdict
from timeit import default_timer as timer
from scipy import stats
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import model, plottools

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(process)d %(processName)s: %(message)s',)
fn = 'sliceCrossingMSD'
springs = [0, 0.1, 1, 10, 25, 100]
kb = 0.0013806488
T = 298
crossing = 0

"""
A worker process to calculate the D(z) for a particular k.
Prototype:
    arg[0]: (double) k - spring constant
    arg[1]: (mp.queue) q - queue to watch
    arg[2]: (int) slice - slice
    arg[3]: (int) randseed - np.seed
"""
def worker(args):
    logging.debug("Reporting for duty: k = %f, slice = %d, seed = %d"%(args[0], args[2], args[3]))
    k = args[0]
    out_q = args[1]
    length = 6*math.pow(10,-8)  # 60 ns
    np.random.seed(args[3])
    dt = 2*math.pow(10,-15)
    traj = model.genTrajC(k,
            mass = 5,
            r = 7.3,
            length=length,
            dt = dt,
            verbosity = 0,
            random = 1)
    logging.debug("Done generating trajectory...")
    slice = args[2] 

    #####################
    # HUMMER 
    ##################### 
    pacr = model.fftAutocorrelation(traj[0][::slice])
    index = np.where(np.diff(np.sign(pacr)))[0]
    if len(index):
        if index[crossing]==0:
            pacrtrunc = pacr[0:1]   # ensure  non-zero length
        else:
            pacrtrunc = pacr[0:index[crossing]]
    else:
        pacrtrunc = pacr
    tau = np.sum(pacrtrunc)*dt*slice
    Dhummer = np.var(traj[0] - np.mean(traj[0]))*1e-16/tau
    logging.debug("Pacr calculation complete")
    
    ####################
    # GREEN-KUBO
    ####################
    vacr = model.fftAutocovariance(traj[1][::slice])
    index = np.where(np.diff(np.sign(vacr)))[0]
    if len(index):
        if index[crossing]==0:
            vacrtrunc = vacr[0:1]   # ensure  non-zero length
        else:
            vacrtrunc = vacr[0:index[crossing]]
    else:
        vacrtrunc = vacr
    Dvacr = np.sum(vacrtrunc)*dt*slice*1e-16
    logging.debug("Vacr calculation complete")
    
    ####################
    # SFDT
    ####################
    facr = model.fftAutocovariance(traj[2][::slice])
    index = np.where(np.diff(np.sign(facr)))[0]
    if len(index):
        if index[crossing] == 0:
            facrtrunc = facr[0:1]
        else:
            facrtrunc = facr[0:index[crossing]]
    else:
        facrtrunc = facr
    Dfacr = 2*(kb*T)**2/(np.sum(facrtrunc)*dt*slice)*10**-16 
    logging.debug("Facr calculation complete")

    ####################
    # MSD
    ####################
    stride = int(25000/slice)# 50ps @2fs ts divided by stride  
    skip = int(2500/slice)
    positions = traj[0][::slice]
    # Discard data at the end in case of mismatch
    if not len(positions)%stride == 0: 
        positions = positions[0:int(len(positions)/stride)*stride]
    diff = np.zeros((len(positions)/stride, stride))
    for i in np.arange(0, len(positions), stride):
        for j in np.arange(0, stride, 1):
            diff[i/stride][j] = positions[i+j] - positions[i] 
    diff = np.power(diff, 2)
    mean = diff.mean(axis=0)
    tsteps = np.arange(0, int(50000)/(slice*2)*slice*2, slice*2)
    tskip = tsteps[skip:]
    mskip = mean[skip:]
    logging.debug(tsteps) 
    logging.debug("%d %d"%(len(tskip), len(mskip)))
    slope, intercept, r_value, p_value, std_err = stats.linregress(tskip, mskip)
    Dmsd = slope/2*0.1 # cm^2/s
    logging.debug("MSD calculation complete")

    #logging.info("%0.1f %0.2f %0.4e %0.4e"%(args[0], args[2], Dmsd, r_value**2))
    logging.info("%0.1f %0.2f %0.4e %0.4e %0.4e %0.4e"%(args[0], args[2], Dhummer, Dvacr, Dfacr, Dmsd))
    out_q.put("%0.1f %0.2f %0.4e %0.4e %0.4e %0.4e"%(args[0], args[2], Dhummer, Dvacr, Dfacr, Dmsd))

"""
A listener to watch the queue and write to a file:
Prototype: 
    arg[0]: (str) fname - filename
    arg[1]: (mp.queue) q - queue to watch
"""
def writer(q):
    while 1:
        m = q.get()
        if m =='kill':
            f.write('killed')
            break
        fd = open('datasets/k%s_%s'%(m.split()[0], fn), 'ab+')
        fd.write(str(m) + '\n')
        fd.flush()
        fd.close()

def run():
    logging.debug("Starting the run")
    manager = multiprocessing.Manager()
    out_q = manager.Queue()
    pool = multiprocessing.Pool(maxtasksperchild=1, processes=16)
    watcher = pool.apply_async(writer, (out_q, ))
    slices = xrange(1, 500, 5)
    #slices = [11]
    jobs = []
    for spring in springs:
        for slice in slices:
            for i in np.arange(0, 100, 1):
                jobs.append((spring, out_q, slice, i))
    logging.debug("Done generating job prototypes")
    pool.map(worker, iter(jobs)) 
    out_q.put('kill')   # Kill the writer
    pool.close()
    pool.join()

def plot():
    matplotlib.rc('font', **{'size':14})
    label_font = {'size':18}

    """
    PLOT THE ANALYSIS FOR THE HUMMER STUFF
    """
    fig1 = plt.figure(1, facecolor='white', figsize=(10,8))
    ax1 = fig1.add_subplot(111)
    ax1.margins(0.025, 0.05)
    inset = plt.axes([0.15, 0.60, 0.25, 0.25])

    for spring in springs:
        data = np.loadtxt('datasets/k%0.1f_%s'%(spring, fn))
        d = defaultdict(np.array)
        stacked = np.array([])
        for row in data:
            row[2] = row[2]*10**6
            if d.has_key(row[1]):
                d[row[1]] = np.append(d[row[1]], row[2])
            else:
                d[row[1]] = np.array(row[2])
        keylist = d.keys()
        keylist.sort()
        for key in keylist:
            if stacked.size == 0:
                stacked = d[key].T
            else:
                stacked = np.column_stack((stacked, d[key].T))
        mean =  stacked.mean(axis=0) 
        stdev = stacked.std(axis=0)
        ax1.plot(keylist, mean, '-', label=r'$k = %s$'%(spring))
        inset.plot(keylist, mean, '-')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_title('Hummer Estimator')
    ax1.set_xlabel(r'Writeout Interval [steps]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    ax1.set_xlim([1,20])
    inset.axhline(y=3.34455, linestyle='--', color='k')
    inset.set_xlabel(r'Writeout Inverval [steps]')
    inset.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/hummer_interval.png', dpi=1200, transparency=True) 
    plt.savefig('Figures/hummer_interval.eps', dpi=1200)

    """
    HUMMER ZOOM 
    """
    fig2 = plt.figure(2, facecolor='white', figsize=(10,8))
    ax1 = fig2.add_subplot(111)
    ax1.margins(0.025, 0.05)
    inset = plt.axes([0.15, 0.68, 0.25, 0.25])

    for spring in springs:
        data = np.loadtxt('datasets/k%0.1f_%s'%(spring, fn))
        d = defaultdict(np.array)
        stacked = np.array([])
        for row in data:
            row[2] = row[2]*10**6
            if d.has_key(row[1]):
                d[row[1]] = np.append(d[row[1]], row[2])
            else:
                d[row[1]] = np.array(row[2])
        keylist = d.keys()
        keylist.sort()
        for key in keylist:
            if stacked.size == 0:
                stacked = d[key].T
            else:
                stacked = np.column_stack((stacked, d[key].T))
        mean =  stacked.mean(axis=0) 
        stdev = stacked.std(axis=0)
        ax1.plot(keylist, mean, '-', label=r'$k = %s$'%(spring))
        inset.plot(keylist, mean, '-')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_title('Hummer Estimator')
    ax1.set_xlabel(r'Writeout Interval [steps]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    ax1.set_xlim([1,20])
    ax1.set_ylim([3,3.6])
    inset.axhline(y=3.34455, linestyle='--', color='k')
    inset.set_xlabel(r'Writeout Inverval [steps]')
    inset.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/hummer_interval_zoom.png', dpi=1200, transparency=True) 
    plt.savefig('Figures/hummer_interval_zoom.eps', dpi=1200) 
    
    """
    PLOT THE ANALYSIS FOR THE GREEN-KUBO STUFF
    """
    fig3 = plt.figure(3, facecolor='white', figsize=(10,8))
    ax1 = fig3.add_subplot(111)
    ax1.margins(0.025, 0.05)
    inset = plt.axes([0.15, 0.65, 0.25, 0.25])
    for spring in springs:
        data = np.loadtxt('datasets/k%0.1f_%s'%(spring, fn))
        d = defaultdict(np.array)
        stacked = np.array([])
        for row in data:
            row[3] = row[3]*10**6
            if d.has_key(row[1]):
                d[row[1]] = np.append(d[row[1]], row[3])
            else:
                d[row[1]] = np.array(row[3])
        keylist = d.keys()
        keylist.sort()
        for key in keylist:
            if stacked.size == 0:
                stacked = d[key].T
            else:
                stacked = np.column_stack((stacked, d[key].T))
        mean =  stacked.mean(axis=0) 
        stdev = stacked.std(axis=0)
        ax1.plot(keylist, mean, '-', label=r'$k = %s$'%(spring))
        inset.plot(keylist, mean, '-')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_title('Green-Kubo Velocity Autocorrelation')
    ax1.set_xlabel(r'Writeout Interval [steps]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    ax1.set_xlim([1,20])
    ax1.set_ylim([0,8])
    inset.axhline(y=3.34455, linestyle='--', color='k')
    inset.set_xlabel(r'Writeout Inverval [steps]')
    inset.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc=4)
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/green-kubo_interval.png', dpi=1200, transparency=True) 
    plt.savefig('Figures/green-kubo_interval.eps', dpi=1200) 
    
    """
    VACR COMPARE LENGTHS
    """
    fig4 = plt.figure(4, facecolor='white', figsize=(10,8))
    ax1 = fig4.add_subplot(111)
    ax1.margins(0.025, 0.05)
    inset = plt.axes([0.20, 0.65, 0.25, 0.25])
    data = np.loadtxt('datasets/k0.0_%s_20'%(fn))
    d = defaultdict(np.array)
    stacked = np.array([])
    for row in data:
        row[3] = row[3]*10**6
        if d.has_key(row[1]):
            d[row[1]] = np.append(d[row[1]], row[3])
        else:
            d[row[1]] = np.array(row[3])
    keylist = d.keys()
    keylist.sort()
    for key in keylist:
        if stacked.size == 0:
            stacked = d[key].T
        else:
            stacked = np.column_stack((stacked, d[key].T))
    mean =  stacked.mean(axis=0) 
    stdev = stacked.std(axis=0)
    ax1.plot(keylist, mean, '-', label=r'20 ns; k=0.0')
    inset.plot(keylist, mean, '-') 
    data = np.loadtxt('datasets/k0.0_%s'%(fn))
    d = defaultdict(np.array)
    stacked = np.array([])
    for row in data:
        row[3] = row[3]*10**6
        if d.has_key(row[1]):
            d[row[1]] = np.append(d[row[1]], row[3])
        else:
            d[row[1]] = np.array(row[3])
    keylist = d.keys()
    keylist.sort()
    for key in keylist:
        if stacked.size == 0:
            stacked = d[key].T
        else:
            stacked = np.column_stack((stacked, d[key].T))
    mean =  stacked.mean(axis=0) 
    stdev = stacked.std(axis=0)
    ax1.plot(keylist, mean, '-', label=r'60 ns; k=0.0')
    inset.plot(keylist, mean, '-') 

    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_title('Green-Kubo Velocity Autocorrelation')
    ax1.set_xlabel(r'Writeout Interval [steps]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    ax1.set_xlim([1,20])
    ax1.set_ylim([3, 6])
    inset.axhline(y=3.34455, linestyle='--', color='k')
    inset.set_xlabel(r'Writeout Inverval [steps]')
    inset.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc=4)
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/green-kubo_interval_length.png', dpi=1200, transparency=True) 
    plt.savefig('Figures/green-kubo_interval_length.eps', dpi=1200) 

    """
    PLOT THE ANALYSIS FOR SFDT
    """
    fig5 = plt.figure(5, facecolor='white', figsize=(10,8))
    ax1 = fig5.add_subplot(111)
    ax1.margins(0.025, 0.05)
    inset = plt.axes([0.15, 0.65, 0.25, 0.25])
    # The noise term is completely separable so they are all the same
    data = np.loadtxt('datasets/k0.0_%s'%(fn))
    d = defaultdict(np.array)
    stacked = np.array([])
    for row in data:
        row[4] = row[4]*10**6
        if d.has_key(row[1]):
            d[row[1]] = np.append(d[row[1]], row[4])
        else:
            d[row[1]] = np.array(row[4])
    keylist = d.keys()
    keylist.sort()
    for key in keylist:
        if stacked.size == 0:
            stacked = d[key].T
        else:
            stacked = np.column_stack((stacked, d[key].T))
    mean =  stacked.mean(axis=0) 
    stdev = stacked.std(axis=0)
    ax1.plot(keylist, mean, '-', linewidth=2, label=r'$k = 0.0$')
    inset.plot(keylist, mean, '-')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_title('SFDT')
    ax1.set_xlabel(r'Writeout Interval [steps]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    ax1.set_xlim([1,20])
    ax1.set_ylim([0,8])
    inset.axhline(y=3.34455, linestyle='--', color='k')
    inset.set_xlabel(r'Writeout Inverval [steps]')
    inset.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/sfdt_interval.png', dpi=1200, transparency=True) 
    plt.savefig('Figures/sfdt_interval.eps', dpi=1200) 

    """
    PLOT THE ANALYSIS FOR MSD
    """
    fig6 = plt.figure(6, facecolor='white', figsize=(10,8))
    ax1 = fig6.add_subplot(111)
    ax1.margins(0.025, 0.05)
    inset = plt.axes([0.20, 0.15, 0.25, 0.25])
    data = np.loadtxt('datasets/k0.0_%s'%(fn))
    d = defaultdict(np.array)
    stacked = np.array([])
    for row in data:
        row[5] = row[5]*10**6
        if d.has_key(row[1]):
            d[row[1]] = np.append(d[row[1]], row[5])
        else:
            d[row[1]] = np.array(row[5])
    keylist = d.keys()
    keylist.sort()
    for key in keylist:
        if stacked.size == 0:
            stacked = d[key].T
        else:
            stacked = np.column_stack((stacked, d[key].T))
    mean =  stacked.mean(axis=0) 
    stdev = stacked.std(axis=0)
    ax1.plot(keylist, mean, '-', linewidth=2, label=r'$k = 0.0$')
    inset.plot(keylist, mean, '-')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_title('MSD')
    ax1.set_xlabel(r'Writeout Interval [steps]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    ax1.set_xlim([1,20])
    inset.axhline(y=3.34455, linestyle='--', color='k')
    inset.set_xlabel(r'Writeout Inverval [steps]')
    inset.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc=4)
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/MSD_interval.png', dpi=1200, transparency=True) 
    plt.savefig('Figures/MSD_interval.eps', dpi=1200) 
    
    fig7 = plt.figure(7, facecolor='white', figsize=(10,8))
    ax1 = fig7.add_subplot(111)
    ax1.margins(0.025, 0.05)
    inset = plt.axes([0.15, 0.65, 0.25, 0.25])
    for spring in springs:
        data = np.loadtxt('datasets/k%0.1f_%s'%(spring, fn))
        d = defaultdict(np.array)
        stacked = np.array([])
        for row in data:
            row[5] = row[5]*10**6
            if d.has_key(row[1]):
                d[row[1]] = np.append(d[row[1]], row[5])
            else:
                d[row[1]] = np.array(row[5])
        keylist = d.keys()
        keylist.sort()
        for key in keylist:
            if stacked.size == 0:
                stacked = d[key].T
            else:
                stacked = np.column_stack((stacked, d[key].T))
        mean =  stacked.mean(axis=0) 
        stdev = stacked.std(axis=0)
        ax1.plot(keylist, mean, '-', label=r'$k = %s$'%(spring))
        inset.plot(keylist, mean, '-')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_title('MSD')
    ax1.set_xlabel(r'Writeout Interval [steps]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    ax1.set_xlim([1,20])
    ax1.set_ylim([0,8])
    inset.axhline(y=3.34455, linestyle='--', color='k')
    inset.set_xlabel(r'Writeout Inverval [steps]')
    inset.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels)
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/MSD_interval.png', dpi=1200, transparency=True) 
    plt.savefig('Figures/MSD_interval.eps', dpi=1200) 
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
    if len(sys.argv) <2 :
        args = parser.parse_args(['-h'])
        exit()
    if args.fn:
        fn = str(args.fn)
    if args.run:
        run()
    if args.plot:
        plot()
