#!/bin/env python2.7
"""
@description: 
@author: Christopher T. Lee (ctlee@ucsd.edu)
LAST TOUCHED: March 4, 2014
HISTORY:
    March 1, 2014 -  Made nice plots
SAMPLE RUN: Codeine like molecule
    ./model.py -k 0.1 -m 299 -r 10 -t 298 -p
"""
import argparse, math, pdb, random, sys 
import numpy as np
import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import traj_tools, plottools
from numbapro import cuda
from numbapro.cudalib import cufft 
from numbapro.cudalib import curand
from timeit import default_timer as timer

# SOME GLOBAL CONSTANTS
kb = 0.0013806488       # kg*A^2*s^-2*K-1
crossing = 5

def fftAutocovarianceGPU(signal):
    """
    GPU based fft calc. This needs to be seriously optimized still
    """
    centered_signal = signal - np.mean(signal)
    zero_padding = np.zeros_like(centered_signal)
    padded_signal = np.concatenate((centered_signal, zero_padding))
    padded_signal = padded_signal.astype(np.complex128)
    cufft.FFTPlan(shape=padded_signal.shape, itype=np.complex128, otype=np.complex128)
    cufft.fft_inplace(padded_signal)    # Run the inplace fft
    padded_signal = (np.abs(padded_signal)**2).astype(np.complex128) # power spectral density
    cufft.ifft_inplace(padded_signal)
    pseudo_autocovariance = padded_signal/len(padded_signal)
    
    mask = np.concatenate((np.ones_like(centered_signal), np.zeros_like(centered_signal)))
    mask = mask.astype(np.complex128)
    cufft.fft_inplace(mask) 
    mask = (np.abs(mask)**2).astype(np.complex128) 
    cufft.ifft_inplace(mask)
    mask = mask/len(mask)

    autocovariance = pseudo_autocovariance/mask
    return np.real(autocovariance[0:len(signal)])

def fftAutocorrelationGPU(signal):
    """
    FFT calculation of the normalized autocorrelation <df(0) - df(t)>/var(f) without wrapping
    """
    autocovariance = fftAutocovarianceGPU(signal)
    variance = autocovariance.flat[0]
    if variance == 0.:
        return np.zeros(autocovariance.shape)
    else:
        return (autocovariance/variance)

def fftAutocovarianceX(signal):
    """
    FFT based calculation of the autocovariance function <df(0) - df(t)> without wrapping
    """
    zero_padding = np.zeros_like(signal)
    padded_signal = np.concatenate((signal, zero_padding))
    ft_signal = np.fft.fft(padded_signal)
    pseudo_autocovariance = np.fft.ifft(np.abs(ft_signal)**2)
    input_domain  = np.ones_like(signal)
    mask = np.concatenate((input_domain, zero_padding))
    ft_mask = np.fft.fft(mask)
    mask_correction_factors = np.fft.ifft(np.abs(ft_mask)**2)
    autocovariance = pseudo_autocovariance/mask_correction_factors
    return np.real(autocovariance[0:len(signal)])


def fftAutocovariance(signal):
    """
    FFT based calculation of the autocovariance function <df(0) - df(t)> without wrapping
    """
    centered_signal = signal - np.mean(signal)
    zero_padding = np.zeros_like(centered_signal)
    padded_signal = np.concatenate((centered_signal, zero_padding))
    ft_signal = np.fft.fft(padded_signal)
    pseudo_autocovariance = np.fft.ifft(np.abs(ft_signal)**2)
    input_domain  = np.ones_like(centered_signal)
    mask = np.concatenate((input_domain, zero_padding))
    ft_mask = np.fft.fft(mask)
    mask_correction_factors = np.fft.ifft(np.abs(ft_mask)**2)
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

def getACR(signal):
    """
    Calculate the autocorrelation of the signal without wrapping. Meaning the tail will have significantly less data (will be statistically noisy). (Deprecated) Use the fft calculation instead. It's MUCH faster.
    """
    centered_signal = signal - np.mean(signal)
    corr = np.correlate(centered_signal,centered_signal, mode='full') # Total correlation function
    corr = corr[corr.size/2:]   # Remove wrapped
    count = np.arange(centered_signal.size, 0, -1, dtype=np.float64)
    return np.divide(corr, count)

def block(timeseries, blocklen):
    """
    Block standard error calculation 
    """
    nblocks = int(timeseries.size/blocklen)
    blocks = np.array_split(timeseries, nblocks)    # break the data into n blocks of approximately even size
    means = np.zeros(nblocks)
    for index in xrange(1, nblocks): 
        means[index] = np.mean(blocks[index])
    stdev = np.std(means)
    return stdev/math.sqrt(nblocks)

def genTrajC(k = 0,              # Spring constant (kg/s^2)
        mass = 0,                           # Mass (kg)
        r = 0,                              # hydrodynamic radius  (A)
        bias = 0,                           # Underlying pmf
        viscosity = 8.94*math.pow(10,-14),  # kg/A/S of water experimenta
        T = 298,                            # Temperature (K) 
        length = 2*math.pow(10, -8),        # Length (s)
        dt = 2*math.pow(10,-15),            # dt (s)
        pos = 0,                            # Initial position (A) 
        verbosity = 0,                      # How much to write out
        random = 1):                        # To random or not to 
    """
    Generate a sample trajectory for later analysis. Calls the C-python accelerated traj-tools library
    """
    # Set some initial conditions
    vel = 0
    force = 0
    k = k*0.694769 # kg*s^-2
    mass = mass*math.pow(10,-3)/(6.022*math.pow(10,23))  # kg
    c = 6*math.pi*viscosity*r # damping coefficient from Stokes' Law kg*s^-1
    b = 1/(1+c*dt/(2*mass))    # unitless
    a = (1 - c*dt/(2*mass))/(1 + c*dt/(2*mass))   # unitless
        
    N = int(math.ceil(length/dt))
    if verbosity >= 1:
        print "Length = " + str(N)
    if verbosity >= 1:
        if k != 0:
            print 'Damping Ratio = ' + str(c/(2*math.sqrt(mass*k)))    # unitless
    noise = math.sqrt(2*c*kb*T*dt)    # Fluctuation-Dissipation Theorem (kg*A*s^-1)
    if verbosity >= 1:
        print "k = " + str(k) + ' kg/s^2'
        print "Mass = " + str(mass)  + " kg"
        print "Temp = " + str(T) + " K"
        print "r = " + str(r) + " A"
        print "Noise = " + str(noise) + " kgA/s"
        print "Damping Coefficient = " + str(c) + " kg/s"
    # Preallocate the arrays for the results 
    positions = np.empty(N, dtype=np.float64)
    springforces = np.empty(N, dtype=np.float64)
    velocities = np.empty(N, dtype=np.float64)
    if not random:
        np.random.seed(12345)   # my favorite number
        noises = np.random.normal(0, noise, size=N)
    else:
        noises = np.random.normal(0, noise, size=N)
        #noises = curand.normal(0, noise, size=N)
    traj_tools.genTraj(positions,
            velocities,
            noises,
            springforces,
            k, mass, c, a, b, pos, dt, N)
    if verbosity >= 1:
        print "===== Done generating sample trajectory! ====="
    forces = springforces + noises/dt # the random forces
    return np.vstack((positions, velocities, forces))

def generateTrajectoryC(k = 0,              # Spring constant (kg/s^2)
        mass = 0,                           # Mass (kg)
        r = 0,                              # hydrodynamic radius  (A)
        bias = 0,                           # Underlying pmf
        viscosity = 8.94*math.pow(10,-14),  # kg/A/S of water
        T = 298,                            # Temperature (K) 
        length = 2*math.pow(10, -8),        # Length (s)
        dt = 2*math.pow(10,-15),            # dt (s)
        pos = 0,                            # Initial position (A) 
        verbosity = 0,                      # How much to write out
        random = 1):                        # To random or not to 
    """
    Generate a sample trajectory for later analysis. Calls the C-python accelerated traj-tools library
    """
    # Set some initial conditions
    vel = 0
    force = 0
    k = k*0.694769 # kg*s^-2
    mass = mass*math.pow(10,-3)/(6.022*math.pow(10,23))  # kg
    c = 6*math.pi*viscosity*r # damping coefficient from Stokes' Law kg*s^-1
    b = 1/(1+c*dt/(2*mass))    # unitless
    a = (1 - c*dt/(2*mass))/(1 + c*dt/(2*mass))   # unitless
        
    N = int(math.ceil(length/dt))
    if verbosity >= 1:
        print "Length = " + str(N)
    if verbosity >= 1:
        if k != 0:
            print 'Damping Ratio = ' + str(c/(2*math.sqrt(mass*k)))    # unitless
    noise = math.sqrt(2*c*kb*T*dt)    # Fluctuation-Dissipation Theorem (kg*A*s^-1)
    if verbosity >= 1:
        print "k = " + str(k) + ' kg/s^2'
        print "Mass = " + str(mass)  + " kg"
        print "Temp = " + str(T) + " K"
        print "r = " + str(r) + " A"
        print "Noise = " + str(noise) + " kgA/s"
        print "Damping Coefficient = " + str(c) + " kg/s"
        # Preallocate the arrays for the results 
    positions = np.empty(N, dtype=np.float64)
    forces = np.empty(N, dtype=np.float64)
    KE = np.empty(N, dtype=np.float64)
    PE = np.empty(N, dtype=np.float64)
    energies = np.empty(N, dtype=np.float64)
    velocities = np.empty(N, dtype=np.float64)
    
    if not random:
        np.random.seed(12345)   # my favorite number
        noises = np.random.normal(0, noise, size=N)
    else:
        noises = np.random.normal(0, noise, size=N)
    """
    else:
        try:
            noises = curand.normal(0, noise, size=N)
        except:
            print 'GPU OUT OF MEM'
            noises = np.random.normal(0, noise, size=N) # If GPU craps out default to slow cpu
    """
    traj_tools.generateTrajectory(positions,
            velocities,
            noises,
            forces,
            KE,
            PE,
            k, mass, c, a, b, pos, dt, N)
    energies = KE + PE
    if verbosity >= 1:
        print "===== Done generating sample trajectory! ====="

    forces = noises/dt # the random forces
    return positions, velocities, forces, noises, KE, PE, energies

def generateTrajectory(k = 0,               # Spring constant (kg/s^2)
        mass = 0,                           # Mass (kg)
        r = 0,                              # hydrodynamic radius  (A)
        bias = 0,                           # Underlying pmf
        viscosity = 8.94*math.pow(10,-14),  # kg/A/S of water
        T = 298,                            # Temperature (K) 
        length = 2*math.pow(10, -8),        # Length (s)
        dt = 2*math.pow(10,-15),            # dt (s)
        pos = 0,                            # Initial position (A) 
        verbosity = 0,                      # How much to write out
        random = 1): 
    """
    Generate a sample trajectory for later analysis
    """
    # Set some initial conditions
    vel = 0
    force = 0
    k = k*0.694769 # kg*s^-2
    mass = mass*math.pow(10,-3)/(6.022*math.pow(10,23))  # kg
    c = 6*math.pi*viscosity*r # damping coefficient from Stokes' Law kg*s^-1
    b = 1/(1+c*dt/(2*mass))    # unitless
    a = (1 - c*dt/(2*mass))/(1 + c*dt/(2*mass))   # unitless
        
    N = int(math.ceil(length/dt))
    if verbosity >= 1:
        print "Length = " + str(N)
    if verbosity >= 1:
        if k != 0:
            print 'Damping Ratio = ' + str(c/(2*math.sqrt(mass*k)))    # unitless
    noise = math.sqrt(2*c*kb*T*dt)    # Fluctuation-Dissipation Theorem (kg*A*s^-1)
    if verbosity >= 1:
        print "k = " + str(k) + ' kg/s^2'
        print "Mass = " + str(mass)  + " kg"
        print "Temp = " + str(T) + " K"
        print "r = " + str(r) + " A"
        print "Noise = " + str(noise) + " kgA/s"
        print "Damping Coefficient = " + str(c) + " kg/s"

    if not random:
        np.random.seed(12345)   # my favorite number
    noises = np.random.normal(0, noise, size=N) 
    
    # Preallocate the arrays for the results 
    positions = np.empty(N, dtype=np.float64)
    #noises = np.empty(N, dtype=np.float64)
    forces = np.empty(N, dtype=np.float64)
    KE = np.empty(N, dtype=np.float64)
    PE = np.empty(N, dtype=np.float64)
    energies = np.empty(N, dtype=np.float64)
    velocities = np.empty(N, dtype=np.float64)

    for i in xrange(0,N):   # interate steps
        positions[i] = pos
        #bump = random.gauss(0,noise)    # Generate the random kick with variance noise
        #noises[i] = bump
        """
        This integrator is an implementation of:
        Gronbech-Jensen, N., Farago, O., Molecular Physics, V111, 8:989-991, (2013)
        """
        pos = pos + b*dt*vel + b*dt*dt*force/(2*mass) + b*dt/(2*mass)*noises[i]    # A
        fnew = -k*pos + bias    # harmonic oscillator + underlying PMF
        forces[i] = fnew        # kg*A*s^-2
        vel = a*vel + dt/(2*mass)*(a*force + fnew) + b/mass*noises[i] # A/s
        force = fnew
        velocities[i] = vel
        if verbosity >= 2:
            print "velocity = " + str(vel) + " A/s"
        KE[i] = .5*mass*vel**2
        PE[i] = .5*k*pos**2
        energies[i] = .5*mass*vel**2 + .5*k*pos**2 # kgA^2/s^2
    if verbosity >= 1:
        print "===== Done generating sample trajectory! ====="
    return positions, velocities, forces, noises, KE, PE, energies

"""
MAIN THREAD OF EXECUTION FOR TESTING. ALSO FOR LEGACY BENCHMARKS.
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A nice model simulation of a harmonically constrained particle diffusing in one dimension using a Langevin thermostat")
    parser.add_argument('-v',
        '--verbosity',
        action='count',
        help = 'Set the ouput verbosity')
    parser.add_argument('-k',
        action = 'store',
        dest = 'k',
        help = 'Spring constant constraining the particle in kcal/mol/A^2')
    parser.add_argument('-m',
        action = 'store',
        dest = 'mass',
        help = 'Mass of the particle in g/mol')
    parser.add_argument('-t',
        action = 'store',
        dest = 'temp',
        help = 'Turn up the noise to temp in K')
    parser.add_argument('-r',
        action = 'store',
        dest = 'r',
        help = 'Hydrodynamic radius of the particle in angstroms')
    parser.add_argument('-p',
        action = 'store_true',
        dest = 'plot',
        help = 'Generate nice plots of the analysis')
    parser.add_argument('-ch',
        action = 'store_true',
        dest = 'chunksize',
        help = 'Chunk the plots to prevent saturating pixels')
    parser.add_argument('-acr',
        action = 'store_true',
        dest = 'acr',
        help = 'Run the positional autcorrelation based analysis')
    parser.add_argument('-b',
        action = 'store_true',
        dest = 'block',
        help = 'generate the block analysis')
    parser.add_argument('-cvp',
        action = 'store_true',
        dest = 'scancolvars',
        help = 'Scan over colvarstrajfreq')
    args = parser.parse_args()
    if len(sys.argv) < 2:
        # Must pass in at least one command line argument or nothing interesting comes out
        args = parser.parse_args(['-h'])
        exit(0)

    # Matplotlib rc
    matplotlib.rc('font', **{'size':14})
    label_font = {'size':18}
    labelx = -0.1

    # Some default values
    length = 2*math.pow(10, -8)
    dt = 2*math.pow(10,-15)
    N = length/dt
    #viscosity = 8.94*math.pow(10,-14)   # experimental 
    viscosity = 3.21e-14    # tip3p
    passedargs = dict()
    # Parse for user specified values
    if args.temp:
        passedargs.update({'T':float(args.temp)}) 
        T = int(args.temp)  # K
    if args.k:  # kcal*mol^-1*A^-2
        passedargs.update({'k':float(args.k)})
        k = float(args.k) 
    if args.mass:  # g/mol 
        passedargs.update({'mass':float(args.mass)})
        m = float(args.mass) 
    if args.r:  # Angstrom
        passedargs.update({'r':float(args.r)})
        r = float(args.r)
        c = 6*math.pi*viscosity*float(args.r) # damping coefficient from Stokes' Law kg*s^-1
        if float(args.r) != 0.0:
            DEinstein = kb*T/c*math.pow(10,-16) # cm^2/s
        else:
            DEinstein = None 
    positions, velocities, forces, noises, KE, PE, energies = generateTrajectoryC(length = length, 
            dt = dt, 
            viscosity = viscosity,
            verbosity = args.verbosity,
            **passedargs)
    t = np.linspace(0, length, num=N, endpoint=False)*math.pow(10,9)
    if args.plot:
        """
        FIGURE 1: Forces, Noises, Position, Velocities
        """
        chunksize = int(N/1000)    # chunksize to average over for the plot
            
        fig1 = plt.figure(1, facecolor='white', figsize=(8,10))
        ax1 = fig1.add_subplot(411)
        ax2 = fig1.add_subplot(412)
        ax3 = fig1.add_subplot(413)
        ax4 = fig1.add_subplot(414)

        # Plot the forces
        if args.chunksize:
            min_env, max_env, xcenters, ycenters = plottools.chunk(t, forces, chunksize)
           
            ax1.fill_between(xcenters, min_env, max_env, color='gray', 
                    edgecolor='none', alpha=0.5) 
            ax1.plot(xcenters, ycenters, '-')
        else:
            ax1.plot(t, forces, '-')    
        ax1.margins(0.025,0.05)
        ax1.set_ylabel(r'Forces [$kgA/s^2$]', **label_font)
        ax1.yaxis.set_label_coords(labelx, 0.5)

        # Plot the noise
        min_env, max_env, xcenters, ycenters = plottools.chunk(t, noises, chunksize)
        ax2.fill_between(xcenters, min_env, max_env, color='gray', 
                edgecolor='none', alpha=0.5) 
        ax2.plot(xcenters, ycenters, '-')
        ax2.margins(0.025,0.05)
        ax2.set_ylabel(r'Noise [$kgA/s$]', **label_font)
        ax2.yaxis.set_label_coords(labelx, 0.5)

        # Plot the position 
        if args.chunksize:
            min_env, max_env, xcenters, ycenters = plottools.chunk(t, positions, chunksize) 
            ax3.fill_between(xcenters, min_env, max_env, color='gray', 
                    edgecolor='none', alpha=0.5) 
            ax3.plot(xcenters, ycenters, '-')
        else:
            ax3.plot(t, positions, '-')
        ax3.margins(0.025,0.05)
        ax3.set_ylabel(r'Position [$A$]', **label_font)
        ax3.yaxis.set_label_coords(labelx, 0.5)
        
        # Plot the velocities
        if args.chunksize:
            min_env, max_env, xcenters, ycenters = plottools.chunk(t, velocities*math.pow(10,-10), chunksize)
            ax4.fill_between(xcenters, min_env, max_env, color='gray',
                    edgecolor='none', alpha=0.5)
            ax4.plot(xcenters, ycenters, '-')
        else:
            ax4.plot(t, velocities*math.pow(10,-10), '-')
        ax4.margins(0.025, 0.05)
        ax4.set_ylabel(r'Velocity [$m/s$]', **label_font)
        ax4.set_xlabel(r'Time [$ns$]', **label_font)
        ax4.yaxis.set_label_coords(labelx, 0.5)
        plt.tight_layout(pad=0.5) 
        
        """
        FIGURE 2: Total Energy, Kinetic Energy, Potential Energy
        """
        fig2 = plt.figure(2, facecolor='white', figsize=(8,10))
        ax1 = fig2.add_subplot(111)
        if args.chunksize:
            tempchunk = chunksize*10    # Increase the chunksize for the energy 
            min_env, max_env, xcenters, ycenters = plottools.chunk(t, energies, tempchunk)
            l1 = ax1.plot(xcenters, ycenters, '-', label='Total Energy')
            min_env, max_env, xcenters, ycenters = plottools.chunk(t, KE, tempchunk)
            l2 = ax1.plot(xcenters, ycenters, '-', label='Kinetic Energy')
            min_env, max_env, xcenters, ycenters = plottools.chunk(t, PE, tempchunk)
            l3 = ax1.plot(xcenters, ycenters, '-', label='Potential Energy')
        else:
            l1 = ax1.plot(t, energies, '-', label='Total Energy')
            l2 = ax1.plot(t, KE, '-', label='Kinetic Energy')
            l3 = ax1.plot(t, PE, '-', label='Potential Energy')
        l4 = ax1.axhline(kb*T, label=r'$k_BT$', linewidth=2, linestyle='--', color='k')
        ax1.margins(0.025, 0.05)
        ax1.set_ylabel(r'Energy [$kgA^2/s^2$]', **label_font)
        ax1.set_xlabel(r'Time [$ns$]', **label_font)
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles, labels)
        plt.tight_layout(pad=0.5)

    if args.verbosity >= 1:
        print '<E> = ' + str(np.mean(energies)) + '; kBT = ' +  str(kb*T)
        print '<z> = ' + str(np.mean(positions)) + '; var(z) = ' + str(np.var(positions))
    
    # Effective colvarstrajfreq 
    slices = [1]                       # colvarstrajfreq default set   
    if args.scancolvars:
        slices = xrange(1, 500, 1) 

    if args.acr:
        """
        Figure 3: Autocorrelation functions 
        """
        fig3, axarr = plt.subplots(2,2, **{'num':3, 'facecolor':'white', 'figsize':(12,8)})
        ax1 = axarr[0,0]
        ax2 = axarr[1,0]
        ax3 = axarr[0,1]
        ax4 = axarr[1,1]
        fig3.subplots_adjust(wspace=0.1, hspace=0.2) 

        ax1.set_ylabel(r'$\langle \delta r(0) \delta r(\tau)\rangle/var(r)$', **label_font)
        ax1.margins(0.025, 0.05)
        #ax1.set_xlim(0, 0.5)
        ax3.margins(0.025, 0.05)
        ax2.margins(0.025, 0.05)
        ax2.set_ylabel(r'$\langle \dot{r}(0) \dot{r}(\tau)\rangle$', **label_font)
        ax2.set_xlabel(r'$\tau$ [$ns$]', **label_font)
        #ax2.set_xlim(0, 0.5)
        
        ax4.margins(0.025, 0.05)
        ax4.set_xlabel(r'$\tau$ [$ns$]', **label_font)
        # Look at the data at varying "checkpoint" intervals
        for slice in slices:
            tslice = t[::slice]
            posslice = positions[::slice]
            velslice = velocities[::slice] 
            pacr = fftAutocorrelation(posslice)    # A^2
            # Calculate the relaxation time tau
            #index = np.where(np.logical_and(acr>=-1*threshold, acr <=threshold))
            index = np.where(np.diff(np.sign(pacr)))[0]
            #print index 
            crossing = -1
            if len(index):
                pacrtrunc = pacr[0:index[crossing]] 
            tau = np.sum(pacrtrunc)*dt*slice   # ACR time (s)
            Dpacr = np.var(posslice - np.mean(posslice))*math.pow(10,-16)/tau # Diffusivity cm^2/s
            if args.plot:
                # Plot the position autocorrelation function
                ax1.plot(tslice, pacr, '-', label=slice)
                ax3.plot(tslice[0:index[crossing]], pacrtrunc, '-', label=slice)
            
            crossing = 5
            vacr = fftAutocovariance(velslice)   # A^2/s^2
            #index = np.where(np.logical_and(vacr>=-1*threshold, vacr <=threshold))[0]
            index = np.where(np.diff(np.sign(vacr)))[0]
            #print index
            if len(index):
                vacrtrunc = vacr[0:index[crossing]]
            D0 = np.sum(vacrtrunc)*dt*slice*math.pow(10,-16) # cm^2/s
            if args.plot:
                # Plot the velocity autocorrelation function
                ax2.plot(tslice, vacr, '-', label=slice)
                ax4.plot(tslice[0:index[crossing]], vacrtrunc, '-', label=slice)
            #data = np.column_stack((tslice.T,acr.T))
            #np.savetxt('acr_step%d_k%s'%(slice, args.k), data)
            #print slice, tau, np.var(posslice - np.mean(posslice))
            print "slice, Dpacr, Dvacr, DEinstein, tau"
            print slice, Dpacr, D0, DEinstein, tau
            
            """
            crossing = 0
            facr = fftAutocovariance(forces)
            index = np.where(np.diff(np.sign(facr)))[0]
            if len(index):
                facrtrunc = facr[0:index[crossing]]
            else:
                facrtrunc = facr
            Dfacr = (kb*T)**2/(np.sum(facrtrunc)*dt*slice)*10**-16 
            print Dfacr
            
            fig4 = plt.figure(4, facecolor='white', figsize=(10,8))
            ax1 = fig4.add_subplot(211)
            ax1.margins(0.025, 0.05)
            ax1.plot(tslice, facr, '-', label=slice)
            ax2 = fig4.add_subplot(212)
            ax2.margins(0.025, 0.05)
            ax2.plot(tslice[0:index[crossing]], facrtrunc, '-', label=slice)
            """
        #handles, labels = ax1.get_legend_handles_labels()
        #ax1.legend(handles, labels)
        #handles, labels = ax2.get_legend_handles_labels()
        #ax2.legend(handles, labels)
    if args.plot:
         plt.tight_layout(pad=0.5)
         plt.show()
