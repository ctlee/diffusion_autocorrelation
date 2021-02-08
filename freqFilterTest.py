#!/bin/env python2.7
"""
@description: lets play with filtering of the fft
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: March 21, 2015
HISTORY:
"""
from __future__ import division
import argparse, os, sys, math
import numpy as np
import scipy as sp
import scipy.signal
import model
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from timeit import default_timer as timer


matplotlib.rc('font', **{'size':14})
label_font = {'size':18}
margins = (0.025, 0.05)

def main():
    # Sampling rate
    fs = 128  # Hz
    
    # Time is from 0 to 1 seconds, but leave off the endpoint, so
    # that 1.0 seconds is the first sample of the *next* chunk
    length = 1   # second
    N = int(fs * length)
    t = np.linspace(0, length-0.25, num=N, endpoint=False)
    
    # Generate a sinusoid at frequency f
    a = np.cos(2*np.pi*2*t)
    b = np.cos(2*np.pi*5*t)
    c = np.cos(2*np.pi*10*t)
    d = np.cos(2*np.pi*26*t)
    signal = a+b+c+d
    #signal = a

    fig1 = plt.figure(1, facecolor='white', figsize=(8, 10))

    # Plot signal, showing how endpoints wrap from one chunk to the next
    ax1 = fig1.add_subplot(4, 1, 1)
    plt.plot(t, a, 'r--')
    ax1.plot(t, signal, '.-')
    ax1.margins(*margins)
    ax1.set_title('Original Signal') 
    ax1.set_xlabel('Time [s]')

    zero_padding  = np.zeros_like(signal)
    padded_signal = np.concatenate((signal, zero_padding)) 

    # Use FFT to get the amplitude of the spectrum
    ampl = 1/(N) * np.fft.fft(signal)
    # FFT frequency bins
    freqs = np.fft.fftfreq(N, 1/fs)
    ax2 = fig1.add_subplot(4, 1, 2)
    ax2.stem(np.fft.fftshift(freqs), np.fft.fftshift(ampl))
    ax2.set_xlim([-30,30])
    ax2.margins(*margins)
    ax2.set_title('FFT of Signal')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlabel('Frequency [Hz]')
  
    # BRICKWALL FILTER
    for i in xrange(len(ampl)):
        if np.abs(freqs[i]) > 3:
            #print ampl[i]
            ampl[i] = 0
    """
    filtered_signal = lowpass(signal, fs, 2)
    ampl = 1/(N) * np.fft.fft(filtered_signal)
    """
    # Plot shifted data on a shifted axis
    ax3 = fig1.add_subplot(4, 1, 3)
    ax3.stem(np.fft.fftshift(freqs), np.fft.fftshift(ampl))
    ax3.set_xlim([-30,30])
    ax3.margins(*margins)
    ax3.set_title('FFT Post Filtering')
    ax3.set_ylabel('Amplitude')
    ax3.set_xlabel('Frequency [Hz]')
    
    result = np.fft.ifft(N*ampl)

    ax4 = fig1.add_subplot(4, 1, 4)
    ax4.plot(t, a, '--r', linewidth=1.25, label='original')
    ax4.plot(t, result, '-', linewidth=1, label='ifft')
    ax4.plot(t, a-result, 'k', label='residual')
    ax4.margins(*margins)
    ax4.set_title('Signal Post-Filtering')
    ax4.set_xlabel(r'Time [s]')
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    
    plt.tight_layout()
    plt.show()

def lowpass(data, samprate, cutoff):
    b,a = sp.signal.butter(2,cutoff/(samprate/2.0), btype='low', analog=0, output='ba')
    data_f = sp.signal.filtfilt(b,a,data)
    return data_f


# BEGIN MAIN
if __name__ == '__main__':
    main()
