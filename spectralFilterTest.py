#!/bin/env python2.7
"""
@description: The effect of frequency cutoff on the diffusivity
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: September 23, 2015
HISTORY:
"""
import argparse, logging, math, multiprocessing, os, sys
import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import model, plottools

matplotlib.rc('font', **{'size':14})
label_font = {'size':18}
margins = (0.025, 0.05)

fn = 'spectralFilterTest.txt'
crossing = 0

"""
A worker process to calculate the D(z) at various filter cutoffs
Prototype:
    args[0]: (mp.queue) q - queueu to watch
    args[1]: (int) cutoff - the cutoff
    args[2]: (np.array) spectral_distribution - spectral distribution
    args[3]: (int) length - length of padded_signal
    args[4]: (float) dt - timestep
    args[5]: (np.array) mask_correction_factors
"""

# TODO move the traj generation to run, just do the cutoff stuff in the workers
def worker(args):
    out_q = args[0]
    cutoff = args[1]*1e-18
    spectral_distribution = args[2]
    dt = args[4]
    freqs = np.fft.fftfreq(args[3], 1/dt)
    mask_correction_factors = args[5]
    # BRICKWALL FILTER
    ## NOTE this is destructive on the spectral_distribution
    for i in xrange(args[3]):
        if np.abs(freqs[i]) > cutoff:
            spectral_distribution[i] = 0            
    pseudo_autocovariance = np.fft.ifft(spectral_distribution)
    autocovariance = pseudo_autocovariance/mask_correction_factors
    acv =  np.real(autocovariance[0:args[3]/2])   # Autocovariance
    acr = acv/acv.flat[0]   # Autocorrelation 
    var = acv.flat[0]       # Variance
    
    index = np.where(np.diff(np.sign(acr)))[0]
    if len(index):
        acrtrunc = acr[0:index[crossing]]
        tau = np.sum(acrtrunc)*dt
        Dfilt = var/tau*1e-16
        out_q.put("%0.4e %0.4e"%(cutoff, Dfilt))
    else:
        print "Poo didn't cross"

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
    pool = multiprocessing.Pool(maxtasksperchild=1, processes=48)
    watcher = pool.apply_async(writer, (out_q, ))
   
    
    for i in np.arange(1,100,1):
        np.random.seed(i) 
        dt = 2e-15
        jobs = []
        traj = model.genTrajC(
                k= 10,
                mass = 299,
                r = 7.3,
                length = 2e-8,
                dt = dt,
                verbosity = 0,
                random = 1)
        signal = traj[0]
        del traj

        # Calculating the positional autocorrelation function
        centered_signal = signal - np.mean(signal)
        zero_padding = np.zeros_like(centered_signal)
        padded_signal = np.concatenate((centered_signal, zero_padding))
        ft_signal = np.fft.fft(padded_signal)

        # Spectral distribution for filtering later
        spectral_distribution = np.abs(ft_signal)**2
        pseudo_autocovariance = np.fft.ifft(spectral_distribution)

        # Generate averaging factors
        input_domain  = np.ones_like(centered_signal)
        mask = np.concatenate((input_domain, zero_padding))
        ft_mask = np.fft.fft(mask)
        mask_correction_factors = np.fft.ifft(np.abs(ft_mask)**2)

        # Average the sums
        autocovariance = pseudo_autocovariance/mask_correction_factors
        
        # Truncate the signal to the deinfed length
        autocovariance = np.real(autocovariance[0:len(signal)])
        variance = autocovariance.flat[0]
        pacr = autocovariance/variance
        
        index = np.where(np.diff(np.sign(pacr)))[0]
        if len(index):
            acrtrunc = pacr[0:index[crossing]]   # An error can arise here!
        tau = np.sum(acrtrunc)*dt
        
        # The unfiltered estimated diffusivity
        Dpacr = np.var(signal - np.mean(signal))*1e-16/tau
        
        for cutoff in xrange(100, 1, -1):
            jobs.append((out_q, cutoff, spectral_distribution, len(padded_signal), dt, mask_correction_factors))
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
    mean = stacked.mean(axis=0)
    
    matplotlib.rc('font', **{'size':14})
    label_font = {'size':18}
    fig1 = plt.figure(1, facecolor='white', figsize=(10,8))
    
    ax1 = fig1.add_subplot(111)
    ax1.margins(0.025, 0.05)
    
    ax1.plot(keylist, mean, 'o', color='k', label='Mean') 
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    #ax1.set_ylim([2.5, 4])
    ax1.set_xlabel('Cutoff frequency [hz]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^{-6}$]')
    plt.tight_layout(pad=0.5)
    plt.savefig('Figures/spectralFilterTest.png', dpi=300, transparency=True)
    plt.show()

# BEGIN MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Take a look at the effect of filtering on the simulation")
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
