# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:33:31 2020

@author: wantysal
"""

#Standard library import
import numpy as np
from numpy.fft import fft,ifft
import cmath


# Local import
from mosqito.functions.roughness.excitation_pattern import spec_excitation

def calc_spec_roughness(spectrum, phase, freq_axis, bark_axis, H):
    """ Specific roughness calculation within 47 overlapping 1-bark-wide intervals. 
        The code is based on the algorithm described in "Psychoacoustical roughness:
        implementation of an optimized model" by Daniel and Weber in 1997.
        It includes 3 successive stages : conversion of the original spectrum in specific 
        excitations functions, modulation depth calculation and specific roughness
        calculation.
    
    Parameters
    ----------
    spectrum : numpy.array
              spectrum of the time frame considered
    phase : numpy.array
            phases of the spectrum 
    freq_axis : numpy.array
               frequency axis in Hz
    bark_axis : numpy.array
                frequency axis in Bark
    H : numpy.array
        weighting functions Hi

    Returns
    -------
    R_spec : numpy.array
            47 specific roughnesses within the considered time frame 

    """       

#------------------------------1st stage---------------------------------------
#----------------Creation of the specific excitations functions----------------


# Excitation pattern within 47 overlapping 1-Bark-wide intervals
    spec_excitation_spectrum = spec_excitation(spectrum, freq_axis, bark_axis)

# Specific excitation spectrum size
    N = spec_excitation_spectrum.shape[1]
    
# Transformation in anticipation of subsequent inverse fourier transform     
    spec_excitation_spectrum = np.power(10,spec_excitation_spectrum/20)*0.00002*int(N/2)
                                  
# Inverse Fourier transform of each of the 47 specific excitation spectra
# yielding 200 ms long time functions ei(t) corresponding to the specific excitations  

    # The original phases of the frame spectrum are used
    complex_spec_excitation_spectrum = np.zeros(spec_excitation_spectrum.shape, dtype=complex)
    for i_bark in range(47):
        for i_freq in range(N):
            complex_spec_excitation_spectrum[i_bark,i_freq] = cmath.rect(spec_excitation_spectrum[i_bark,i_freq],phase[i_freq])
                
    spec_excitation_function = np.real(ifft(complex_spec_excitation_spectrum))
       
        
#-------------------------------2d stage---------------------------------------
#---------------------modulation depth calculation----------------------------- 


# The fluctuations of the envelope are contained in the low frequency part 
# of the spectrum of specific excitations in absolute value :      
    envelope_fourier = fft(np.absolute(spec_excitation_function))    

# The real part of the spectrum is appropriately weighted in order to model 
# this low-frequency bandpass characteristic of the roughness on modulation frequency
    envelope_spectrum = np.real(envelope_fourier)  
    weighted_envelope_real =  np.zeros((47,N))      
    weighted_envelope_real = envelope_spectrum * H

      
# The original arguments of the spectrum are used to recreate a complex spectrum after weighting
    envelope_phase = np.imag(envelope_fourier)    
    weighted_envelope = np.zeros((47,N), dtype=complex) 
    for i_bark in range(47):
        for i_freq in range(N):
            weighted_envelope[i_bark,i_freq] = weighted_envelope_real[i_bark,i_freq] + envelope_phase[i_bark,i_freq] * 1j
            
# The time functions of the bandpass filtered envelopes hBPi(t) 
# are calculated via inverse Fourier transform :          
    hBP = np.real(ifft(weighted_envelope))


# Excitation envelope RMS value and temporal average calculation
# followed by the modulation depth estimation
    RMSh = np.zeros((47))
    h0 = np.zeros((47))
    m = np.zeros((47))
    
    for i in range(47):
        RMSh[i] = np.sqrt(np.mean(np.power(hBP[i,:],2)))
        h0[i] = np.mean(np.absolute(spec_excitation_function[i,:]))
        if h0[i]>0:
            if (RMSh[i]/h0[i])<=1:
                m[i] = (RMSh[i]/h0[i])
            elif (RMSh[i]/h0[i])>1:
                m[i] = 1
        else :
            m[i] = 0
    
#-------------------------------3rd stage---------------------------------------
#-----------------------Specific roughness calculation--------------------------

# Modulation depth weighting function given by Aures
    G = np.array([0.5, 0.5, 0.55, 0.58, 0.6, 0.64, 0.68, 0.7, 0.74, 0.76, 0.79, 0.81,
               0.84, 0.89, 0.95, 1.01, 1.07, 1.1, 1.13, 1.15, 1.17, 1.16, 1.15,
               1.13, 1.1, 1.06, 1.02, 0.98, 0.93, 0.88, 0.83, 0.8, 0.78, 0.76,
               0.74, 0.72, 0.71, 0.70, 0.69, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7 ])

# Crosscorrelation coefficients ki2 between the envelopes of the channels i-2 and i
# and ki calculated from channels i and i+2 with dz= 1 bark    
    ki2 = np.zeros((47))
    ki = np.zeros((47))


    for i in range(2,47):
        if hBP[i-2].all()==0 or hBP[i].all()==0:
            ki2[i]=1
        else:
            ki2[i] = np.corrcoef(hBP[i-2,:],hBP[i,:])[0,1]
    for i in range(0,45):   
        if hBP[i].all()==0 or hBP[i+2].all()==0:
           ki[i]=1 
        else:
            ki[i] = np.corrcoef(hBP[i,:],hBP[i+2,:])[0,1]
    ki2[0] = 1    
    ki2[1] = 1
    ki[45] = 1
    ki[46] = 1


# Specific roughness calculation

    R_spec = np.zeros((47))
    
    for i in range(47):
        R_spec[i] = pow(G[i]*m[i]*ki2[i]*ki[i],2)
        
    return R_spec