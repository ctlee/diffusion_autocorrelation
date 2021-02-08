#!/bin/env python2.7
"""
@description: lets play with filtering of the fft
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: March 21, 2015
HISTORY:
"""
from __future__ import division
import argparse, os, sys, threading
import numpy as np
import model
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

def main():
    """
    traj = model.genTrajC(k = 10,
            mass = 299,
            r = 10,
            length=2*10**-8,
            dt = 2*math.pow(10,-15),
            verbosity = 0,
            random = 1)
    pacr = fftAutocovariance(traj[0])
    """
    # Sampling rate
    fs = 32 # Hz
    
    # Time is from 0 to 1 seconds, but leave off the endpoint, so
    # that 1.0 seconds is the first sample of the *next* chunk
    length = 1  # second
    N = fs * length 
    t = np.linspace(0, length, num=N, endpoint=False)
    
    # Generate a sinusoid at frequency f
    f = 1  # Hz
    a = np.cos(2 * np.pi * f * t)
    
    # Plot signal, showing how endpoints wrap from one chunk to the next
    plt.subplot(3, 1, 1)
    plt.plot(t, a, '.-')
    plt.plot(1, 1, 'r.')  # first sample of next chunk
    plt.margins(0.01, 0.1)
    plt.xlabel('Time [s]')

    # Use FFT to get the amplitude of the spectrum
    ampl = 1/N * np.absolute(np.fft.fft(a))
    
    # FFT frequency bins
    freqs = np.fft.fftfreq(N, 1/fs)
    print freqs
    print ampl

    # Plot shifted data on a shifted axis
    plt.subplot(3, 1, 2)
    plt.stem(np.fft.fftshift(freqs), np.fft.fftshift(ampl))
    plt.margins(0.1, 0.1)
    
    plt.subplot(3,1,3)
    plt.plot(t, np.fft.ifft(np.fft.fft(a)) , '.-')
    plt.margins(0.025, 0.05)
    plt.xlabel('Frequency [Hz]')
    
    plt.show()

def fftAutocovariance(signal):
    """
    FFT based calculation of the autocovariance function <df(0) - df(t)> without wrapping
    """
    centered_signal = signal - np.mean(signal)
    zero_padding = np.zeros_like(centered_signal)
    padded_signal = np.concatenate((centered_signal, zero_padding))
    
    ft_signal = np.fft.rfft(_signal)

    n = ft_signal.size
    timestep = 0.1
    freq = np.fft.fftfreq(n, d=timestep)

    fig2 = plt.figure(2, facecolor='white', figsize=(10,10))
    ax1 = fig2.add_subplot(111)
    ax1.plot(freq, ft_signal)
     
    pseudo_autocovariance = np.fft.irfft(np.abs(ft_signal)**2)
    input_domain  = np.ones_like(centered_signal)
    mask = np.concatenate((input_domain, zero_padding))
    ft_mask = np.fft.rfft(mask)
    mask_correction_factors = np.fft.irfft(np.abs(ft_mask)**2)
    autocovariance = pseudo_autocovariance/mask_correction_factors
    return np.real(autocovariance[0:len(signal)])

def fftAutocorrelation(signal):
    """
    FFT calculation of the normalized autocorrelation <df(0) - df(t)>/var(f) without wrapping
    """
    autocovariance = fftAutocovariance(signal)
    variance = autocovariance.flat[0]
    if variance == 0.:
        return np.zeros(autocovariance.shape)
    else:
        return (autocovariance/variance)

# BEGIN MAIN
if __name__ == '__main__':
    main()
