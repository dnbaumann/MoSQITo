# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:29:47 2020

@author: wantysal
"""
import sys
sys.path.append('../../..')

# Standard library import
import numpy as np
from numpy.fft import fft, fftfreq
from math import floor

# Local imports
from mosqito.functions.generic.conversion_bark2frequency import freq2bark
from mosqito.functions.generic.a0_zwicker import a0_definition
from mosqito.functions.roughness.H_function import H_function
from mosqito.functions.roughness.specific_roughness import calc_spec_roughness

def calc_roughness(signal, fs, overlap):
    """ Roughness calculation of a signal sampled at 48kHz.

    The code is based on the algorithm described in "Psychoacoustical roughness:
    implementation of an optimized model" by Daniel and Weber in 1997.
    The roughness model consists of a parallel processing structure that is made up 
    of successive stages and calculates intermediate specific roughnesses R_spec, 
    which are summed up to determine the total roughness R.
    
    Parameters
    ----------
    signal : numpy.array
             signal amplitude values along time
    fs : integer
        sampling frequency
    overlap : float
              overlapping coefficient for the time windows of 200ms 

    Outputs
    -------
    R : numpy.array
        roughness
    R_spec : numpy.array
           specific roughness value within each critical band    
    
    """
  

# Sampling frequency check
    if fs != 48000:
        raise ValueError("ERROR: Sampling frequency must be 48 kHz.")

#  Creation of overlapping 200 ms frames of the sampled input signal 

    # Number of sample points within each frame
    N = int(0.2*fs)
      
    
    # Signal cutting according to the time resolution of 0.2s
    # with the given overlap proportion (number of rows = number of frames)

    row = floor(signal.size/((1-overlap)*N))-1  
    reshaped_signal = np.zeros((row,N))   
    
    
    for i_row in range(row):        
        reshaped_signal[i_row,:] = signal[i_row*int(N*(1-overlap)):i_row*int(N*(1-overlap))+N]       
    
# Signal spectrum is created with a Blackman window for each 200ms time period
    fourier = fft(reshaped_signal* np.blackman(N))
    phase = np.angle(fourier)
    spectrum = np.absolute(fourier/(int(N/2)*0.00002))[:,0:int(N/2)]
    spectrum = 20 * np.log10(spectrum)

    # Frequency axis    
    freq_axis = np.linspace(0,int(fs/2),spectrum.shape[1])
    
    # Conversion of the frequencies into bark values
    bark_axis = freq2bark(freq_axis)
    
# Application of the Zwicker a0 factor representing the transmission between 
# free field and hearing system
    A0 = a0_definition(bark_axis)
    spectrum = spectrum + A0
    			


### The roughness calculation is done within each time frame to compute
### the values along time : 
    print('Roughness is being calculated...')
    
    # H weighting functions definition
    H = H_function(int(N/2))
    
        
    R_spec = np.zeros((spectrum.shape[0],47))
    R = np.zeros((spectrum.shape[0]))

    for i_time in range (spectrum.shape[0]):
        R_spec[i_time,:] = calc_spec_roughness(spectrum[i_time,:], phase[i_time,:],freq_axis,bark_axis,H)
        R[i_time] = 0.25 * sum(R_spec[i_time,:])
        

    return R, R_spec
