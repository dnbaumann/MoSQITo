# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 16:59:53 2020

@author: pc
"""

import numpy as np
from scipy.integrate import simps

def calc_sharpness_din(N, N_specific, is_stationary):
    """ Sharpness calculation

    The code is based on DIN 45692:2009 to determine sharpness
    from specific zwicker loudness.
    
    Parameters
    ----------
    N : float
        loudness value
    N_specific : np.ndarray
        specific critical bands loudness
    is_stationary : boolean
        indicates if the signal is stationary or time-varying
    time : np.array
        time axis

    Outputs
    -------
    S : sharpness
    """
    z = np.linspace(0.1, 24, int(24 / 0.1))
    if is_stationary :
        f = np.zeros((z.size))
        f[z<15.8] = 1
        f[z>=15.8] = N_specific[z>=15.8] *( 0.15 * np.exp(0.42 *( z[z>=15.8] - 15.8)) + 0.85 ) * z[z>=15.8]
        S = (0.11 / N) * simps (f,z)   
        print("DIN sharpness:",str(S),"acum")
    else :
        S = np.zeros((N.size))
        f = np.zeros((z.size,N.size))
        for t in range(N.size):
            f[z<15.8,t] = 1
            f[z>=15.8,t] = N_specific[z>=15.8,t] *( 0.15 * np.exp(0.42 *( z[z>=15.8] - 15.8)) + 0.85 ) * z[z>=15.8] 
            if N[t] >= 1:
                S[t] = (0.11 * simps (f[:,t],z)) / N[t]
    return S