# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 2020

@author: wantysal
"""

# Standard library imports
import numpy as np  

# Local import
from mosqito.functions.sharpness.fastl_g_weighting import g_ZwickerFastl
 
def calc_sharpness_fastl(N, N_specific, is_stationary):
    """ Sharpness calculation according to FASTL's method (1991)
        Expression for weighting function obtained by fitting an 
        equation to the data given in 'Psychoacoustics: Facts and Models' 
        
    Parameters
    ----------
    N : float
        loudness value
    N_specific : np.ndarray
        specific critical bands loudness
    is_stationary : boolean
        indicates if the signal is stationary or time-varying
        """ 
    # Bark axis    
    z = np.linspace(0.1, 24, int(24 / 0.1)) 
    
    # Zwicker and Fastl weighting function
    gZF = g_ZwickerFastl(z)
    
    if is_stationary == True :
        if N == 0:
            S = 0
        else:
            S = 0.11 * sum( N_specific * gZF * z * 0.1) / N
            print("Fastl sharpness:",str(S),"acum")
    else :
        S = np.zeros((N.size))
        for t in range(N.size):
            if N[t] >= 0.1:                
                S[t] = 0.11 * sum(N_specific[:,t] * gZF * z * 0.1) / N[t]
                           
    return S

