# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 14:30:31 2020

@author: pc
"""


import numpy as np

def spec_excitation(spectrum, freq_axis, bark_axis):
    """ Transformation of the frame spectrum into specific excitations spectra
    
According to the article, the excitation level can be converted into triangular 
excitation patterns : the level is set to the corresponding fourier component,
and the excitation decrease slopes are chosen in accordance with the proposal 
of Terhardt :
    S1(f) = -27 dB/Bark for the lower slopes,
    S2(f) = -24-(230/f)+0.2*L dB/Bark for the upper slopes
(f, L respectively the frequency and level of the spectral component concerned)

From that are created specific excitation spectra within 47 overlapping 1-Bark-wide 
intervals with equally spaced centres  zi = 0.5*i Bark (i=1,2,...,47)

                                
    Parameters
    ------------
    spectrum: numpy.array
              signal spectrum of the 200ms frame concerned
    freq_axis: numpy.array
              frequency axis in Hertz
    bark_axis: numpy.array
              frequency axis in Bark

    Output
    ---------
    spec_excitation_spectrum : numpy.array
                     Specific excitation spectrum within 47 1-bark-wide intervals
    
    """
   
# Intervals center frequencies
    center_freq = 0.5 * np.arange(1,48,1)
          
  
# Threshold in quiet in each channel:
# (omission of the contributions whose level is lower than LTQ)
    LTQ = np.array([30.5,18.5,18.5,12.5,12.5,8.5,7.5,7.5,6.5,6.5,5.5,5.5,4.5,4.5,4.5,
                    3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,3.5,4.6,4.6,4.6,4.6,6.2,6.2,6.2,
                    8.4,8.4,8.4,8.6,8.6,7,7,7,4.5,4.5,1,1,0,0,0])
    
# Specific excitation spectra : each frequency component is related to 
# its excitation spectrum  
   
    spec_excitation_spectrum = np.zeros((47,2*spectrum.size))

    # The excitation contributions in each interval [zi - 0.5, zi + 0.5] 
    # are linearly superimposed.
    # The contribution of a spectral component in each interval is:    
    # considering z(f) the frequency of the component given in Bark 
    # If  z(f) > zi + 0.5 Bark : 
    #     contribution to the specific excitation =  S1(Zi + 0.5 Bark)
    # If z(f) < zi - 0.5 Bark : 
    #     contribution to the specific excitation =  S2(Zi - 0.5 Bark)
    # If  z(f) falls into the interval [zi - 0.5; zi + 0.5] : 
    #     contribution to the specific excitation =  L 


    for i in range(0,47):
        for i_freq in range(0,spectrum.size):
            L = spectrum[i_freq]
                
            if bark_axis[i_freq]>=(center_freq[i]+0.5):
                S1 = L - 27 * ( bark_axis[i_freq] - center_freq[i]+0.5 )
                if S1 > LTQ[i]:
                    spec_excitation_spectrum[i,i_freq] = S1
                        
            elif bark_axis[i_freq]<=(center_freq[i]-0.5):
                if freq_axis[i_freq] != 0:
                    S2 = L + (-24 - (230/freq_axis[i_freq] ) + 0.2 * L ) * ( center_freq[i]-0.5 - bark_axis[i_freq])
                    if S2 > LTQ[i]:
                        spec_excitation_spectrum[i,i_freq] = S2
            else :
                if L > LTQ[i]:
                    spec_excitation_spectrum[i,i_freq] = L
                    
            # symmetry in anticipation of a Fourier inverse transform
            spec_excitation_spectrum[i,2*spectrum.size-i_freq-1] = spec_excitation_spectrum[i,i_freq]
                    
    return spec_excitation_spectrum
