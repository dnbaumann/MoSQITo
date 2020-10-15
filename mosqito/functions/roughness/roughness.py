# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 14:29:47 2020

@author: pc
"""
import sys
sys.path.append('../../..')

import numpy as np
from numpy.fft import fft
from math import ceil

from mosqito.functions.generic.conversion_bark2frequency import freq2bark
from mosqito.functions.generic.a0_zwicker import a0_definition
from mosqito.functions.roughness.H_function import H_function
from mosqito.functions.roughness.specific_roughness import calc_spec_roughness

def calc_roughness(signal, fs):
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
    
    # Adaptation of the signal duration to the time resolution
    if (len(signal)/fs)%0.2 != 0 :
        signal = np.concatenate((signal,np.zeros((ceil(len(signal)/N) * N ) - len(signal))))   
    
    # Signal cutting according to the time resolution of 0.2s
    # with an overlap of 0.1s (number of rows = number of frames)

    row = int(2*signal.size/N)-1    
    reshaped_signal = np.zeros((row,N))   
    
    for i_row in range(row):
        reshaped_signal[i_row,:] = signal[int(i_row*(N/2)):int(i_row*(N/2)+N)]       
    
# Signal spectrum is created with a Blackman window for each 200ms time period
    fourier = fft(reshaped_signal* np.blackman(N))
    phase = np.angle(fourier)
    spectrum = np.absolute(fourier/N)[:,0:int(N/2)]
    spectrum = 20 * np.log10(spectrum/0.00002)

    # Frequency axis    
    freq_axis = np.arange(int(N/2))*fs/N
    
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
