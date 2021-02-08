#!/bin/env python2.7
"""
@description: The effect of frequency cutoff on the diffusivity
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: March 21, 2015
HISTORY:
"""
import argparse, os, sys, math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.ndimage.filters
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import model, plottools

matplotlib.rc('font', **{'size':14})
label_font = {'size':18}
margins = (0.025, 0.05)

def main():
    k = 10
    length = 2e-8
    dt = 2e-15
    traj = model.genTrajC(k=k,
            mass = 299,
            r = 7.3,
            length = length,
            dt = dt,
            verbosity = 0,
            random = 0)
    signal = traj[0]
    del traj
  
    """
    # BUTTERWORTH FILTER
    b,a = sp.signal.butter(2, .3e-18/(dt/2.0), btype='low', analog=0, output='ba')
    f_signal = sp.signal.filtfilt(b,a,signal)
    #f_signal = sp.ndimage.filters.gaussian_filter(signal, 1000, order=0)
    """

    t = np.linspace(0, length, length/dt, endpoint=False)*1e9
    """
    plt.figure(1, facecolor='white', figsize=(10,8)) 
    plt.subplot(2,1,1)
    plt.plot(t, signal)
    """
    """
    plt.subplot(2,1,2)
    plt.plot(t, f_signal)
    """ 
    #plt.figure(2, facecolor='white', figsize=(10,8))

    centered_signal = signal - np.mean(signal)
    zero_padding = np.zeros_like(signal)
    padded_signal = np.concatenate((centered_signal, zero_padding))
    ft_signal = np.fft.fft(padded_signal)
    psd = np.abs(ft_signal)**2
    pseudo_autocovariance = np.fft.ifft(psd)
    
    input_domain  = np.ones_like(signal)
    mask = np.concatenate((input_domain, zero_padding))
   
    ft_mask = np.fft.fft(mask)
    mask_correction_factors = np.fft.ifft(np.abs(ft_mask)**2)
    autocovariance = pseudo_autocovariance/mask_correction_factors
    acv =  np.real(autocovariance[0:len(signal)])
    acr = acv/acv.flat[0]  
   
    index = np.where(acr < 0)
    if index[0].size:
        acrtrunc = acr[0:index[0][0]]   # An error can arise here!
    tau = np.sum(acrtrunc)*dt
    D = np.var(signal - np.mean(signal))*1e-16/tau

    # PLOT THE ORIGINAL PSD
    freqs = np.fft.fftfreq(len(padded_signal), 1/dt)
    """
    plt.subplot(4,1,1)
    plt.plot(freqs, psd)
    plt.xlim(-1e-18,1e-18)
    # PLOT THE UNFILTERED ACR
    plt.subplot(4,1,2)
    plt.plot(t,acr)
    #plt.xlim(0,0.5)
    plt.ylim(-1,1)
    """

    cutoff = []
    Dlist = []
    for j in xrange(100, 1, -5):
        # BRICKWALL FILTER
        for i in xrange(len(padded_signal)):
            if np.abs(freqs[i]) > j*1e-18:
                psd[i] = 0            
        """ 
        fc_signal = f_signal - np.mean(f_signal)
        fcp_signal = np.concatenate((fc_signal, zero_padding))
        ftfcp_signal = np.fft.fft(fcp_signal)
        psd = np.abs(ftfcp_signal)**2

        plt.subplot(4,1,3)
        plt.plot(freqs, psd)
        plt.xlim(-1e-18,1e-18)
        """

        pseudo_autocovariance = np.fft.ifft(psd)
        autocovariance = pseudo_autocovariance/mask_correction_factors
        acv =  np.real(autocovariance[0:len(signal)])
        acr = acv/acv.flat[0]  
        var = acv.flat[0]
        """
        plt.subplot(4,1,4)
        plt.plot(t, acr)
        #plt.xlim(0, 0.5)
        plt.ylim(-1,1)
        """ 
        index = np.where(acr < 0)
        if index[0].size:
            acrtrunc = acr[0:index[0][0]]
        tau = np.sum(acrtrunc)*dt
        Dfilttrunc = var/tau*1e-16
        cutoff.append(j*1e-18)
        Dlist.append(Dfilttrunc*10**6)
        print j, Dfilttrunc*10**6
    
    fig1 = plt.figure(1, facecolor='white', figsize=(10,10))
    ax1 = fig1.add_subplot(111)
    ax1.margins(0.025, 0.05)
    ax1.plot(cutoff, Dlist, '.')
    ax1.axhline(y=3.34455, linewidth=2, linestyle='--', color='k', label='Stokes-Einstein')
    ax1.set_xlabel('Cutoff frequency [hz]')
    ax1.set_ylabel(r'$D(z)$ [$cm^2/s\times 10^6$]')
    plt.show()

# BEGIN MAIN
if __name__ == '__main__':
    main()
