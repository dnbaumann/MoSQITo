# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 12:22:23 2020

@author: pc
"""
import numpy as np
from scipy.integrate import simps

def calc_sharpness_aures(N, N_specific, is_stationary):
    """ Sharpness calculation

    The code is based on W. Aures' equation to determine sharpness
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
    if is_stationary == True :
        f = np.zeros((z.size))
        f = N_specific * 0.078 * (np.exp(0.171 * z)) * ( N / np.log(N * 0.05 + 1)) 
        S = (0.11 / N) * simps (f,z)   
        print("Aures sharpness:",str(S),"acum")
    else :
        S = np.zeros((N.size))
        f = np.zeros((z.size,N.size))
        for t in range(N.size):
            if N[t] >= 1:
                f[:,t] += N_specific[:,t]  * 0.078 * (np.exp(0.171 * z)/z) * ( N[t] / np.log(N[t] * 0.05 + 1))  *z  
                S[t] = (0.11 / N[t]) * simps (f[:,t],z)
    return S


