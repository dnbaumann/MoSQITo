# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:06:11 2020

@author: pc
"""

import numpy as np
from scipy.integrate import simps

def calc_sharpness_bismarck(N, N_specific, is_stationary):
    """ Sharpness calculation

    The code is based on Bismarck formulation to determine sharpness
    from specific zwicker loudness. It is quite similar to DIN 45692:2009.
    
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
        f[z<15] = 1
        f[z>=15] = N_specific[z>=15] *( 0.2 * np.exp(0.308 *( z[z>=15] - 15)) + 0.8 ) * z[z>=15]
        S = (0.11 / N) * simps (f,z)   
        print("Bismarck sharpness:",str(S),"acum")
    else :
        S = np.zeros((N.size))
        f = np.zeros((z.size,N.size))
        for t in range(N.size):
            f[z<15,t] = 1
            f[z>=15,t] = N_specific[z>=15,t] *( 0.2 * np.exp(0.308 *( z[z>=15] - 15)) + 0.8 ) * z[z>=15]
            if N[t] >= 1 :
                S[t] = (0.11 / N[t]) * simps (f[:,t],z)
    return S