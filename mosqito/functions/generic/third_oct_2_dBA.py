# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 15:47:08 2020

@author: pc
"""

import numpy as np

def A_weighting(spec_third):
    """ A_weighting dB ponderation according to CEI 61672:2014 """
    
    #ponderation coefficients from the standard
    A_pond = np.array ([ -44.7, 
                 -39.4,
                 -34.6,
                 -30.2,
                 -26.2,
                 -22.5,
                 -19.1,
                 -16.1,
                 -13.4,
                 -10.9,
                 -8.6,
                 -6.6,
                 -4.8,
                 -3.2,
                 -1.9,
                 -0.8,
                 0,
                 0.6,
                 1,
                 1.2,
                 1.3,
                 1.2,
                 1,
                 0.5,
                 -0.1,
                 -1.1,
                 -2.5,
                 -4.3
                 ])
    spec_dBA = np.zeros(spec_third.size)
    for i in range (spec_third.size):
        spec_dBA[i] = spec_third[i] + A_pond[i]
        
    return spec_dBA