# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:33:31 2020

@author: pc
"""

#Standard library import
import numpy as np
from numpy.fft import fft,ifft
import cmath
import scipy.signal as sp

# Local import
from mosqito.functions.roughness.excitation_pattern import spec_excitation

def calc_spec_roughness(spectrum, phase, freq_axis, bark_axis, H):
    """ Specific roughness calculation within 47 overlapping 1-bark-wide intervals. 
        The implemented method is the one described by Daniel and Weber, consisting
        of 3 successive stages : conversion of the original spectrum in specific 
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
                                  
# Inverse Fourier transform of each of the 47 specific excitation spectra
# yielding 200 ms long time functions ei(t) corresponding to the specific excitations  

    # The original phases of the frame spectrum are used
    c_spec_excitation_spectrum = np.zeros(spec_excitation_spectrum.shape, dtype=complex)
    for i_bark in range(47):
        for i_freq in range(N):
            c_spec_excitation_spectrum[i_bark,i_freq] = cmath.rect(spec_excitation_spectrum[i_bark,i_freq],phase[i_freq])

    spec_excitation_function = np.real(ifft(c_spec_excitation_spectrum))
       
        
#-------------------------------2d stage---------------------------------------
#---------------------modulation depth calculation----------------------------- 


# The fluctuations of the envelope are contained in the low frequency part 
# of the spectrum of specific excitations in absolute value       
    envelope = fft(np.absolute(spec_excitation_function[:,:]))[:,0:int(N/2)]
    nangle = np.angle(envelope)

# The previous spectrum is appropriately weighted in order to model the bandpass
# characteristic of the roughness on modulation frequency and symmetryzed in anticipation of IFFT 
    weighted_sym_envelope =  np.zeros((47,N))      
    weighted_sym_envelope[:,0:int(N/2)] = weighted_sym_envelope[:,np.arange(N-1,int((N/2)-1),-1)] = np.multiply(np.abs(envelope),H)
   

# The time functions of the bandpass filtered envelopes hBPi(t) 
# are calculated via inverse Fourier transform and correlated with the envelopes
# hBP,i-2(t) and hBP,i+2(t) of the intervals i-2 and i+2
      
    # Inverse Fourier transform  
    # the original phases from the frame spectrum are used
    weighted_envelope = np.zeros((47,N), dtype=complex) 
    for i_bark in range(47):
        for i_freq in range(int(N/2)):
            weighted_envelope[i_bark,i_freq] = cmath.rect(weighted_sym_envelope[i_bark,i_freq],nangle[i_bark,i_freq])
    hBP = np.real(ifft(weighted_envelope))
 
    
    
# "If there is nonsynchronous random modulation in two or more adjacent critical 
# bands (as with random noise), there is no roughness perception.
# If a very broadband roughness is perceived (like a uniformly amplitude-modulated 
# broadband noise), modulation is synchronous in multiple critical bands."


# Excitation envelope RMS value and temporal average calculation
# followed by the modulation depth estimation
    RMSh = np.zeros((47))
    h0 = np.zeros((47))
    m = np.zeros((47))
    
    for i in range(47):
        RMSh[i] = np.sqrt(np.mean(pow(hBP[i],2)))
        h0[i] = np.mean(np.absolute(spec_excitation_function[i]))
        if h0[i]>0:
            if (RMSh[i]/h0[i])<=1:
                m[i] = (RMSh[i]/h0[i])
            elif (RMSh[i]/h0[i])>1:
                m[i] = 1
        else :
            m[i] = 0
    
#-------------------------------3rd stage---------------------------------------
#-----------------------Specific roughness calculation--------------------------

# Modulation depth weighting function
    G = np.array([0.5, 0.5, 0.55, 0.58, 0.6, 0.64, 0.68, 0.7, 0.74, 0.76, 0.79, 0.81,
               0.84, 0.89, 0.95, 1.01, 1.07, 1.1, 1.13, 1.15, 1.17, 1.16, 1.15,
               1.13, 1.1, 1.06, 1.02, 0.98, 0.93, 0.88, 0.83, 0.8, 0.78, 0.76,
               0.74, 0.72, 0.71, 0.70, 0.69, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7 ])

# Crosscorrelation coefficients ki2 between the envelopes of the channels i-2 and i
# and ki calculated from channels i and i+2 with dz= 1 bark    
    ki2 = np.zeros((47))
    ki = np.zeros((47))
    
    def cross_correlation(x,y):
        n = x.size
        if x.all()==0 and y.all()==0:
            cc = 1
        else:
            cc=(sum(x*y)-(1/n)*sum(x)*sum(y))/np.sqrt((sum(pow(x,2))-(1/n)*pow(sum(x),2))*((sum(pow(y,2)))-(1/n)*(pow(sum(y),2))))      
        return cc

    for i in range(2,47):
        ki2[i] = cross_correlation(spec_excitation_function[i-2,:],spec_excitation_function[i,:])
    for i in range(0,45):    
        ki[i] = cross_correlation(spec_excitation_function[i,:],spec_excitation_function[i+2,:]) 
    ki2[0] = 1    
    ki2[1] = 1
    ki[45] = 1
    ki[46] = 1
    ki[np.isnan(ki)] =0
    ki2[np.isnan(ki2)] =0

# Specific roughness calculation

    R_spec = np.zeros((47))
    
    for i in range(47):
        R_spec[i] = pow(G[i]*m[i]*ki2[i]*ki[i],2)
        
    return R_spec