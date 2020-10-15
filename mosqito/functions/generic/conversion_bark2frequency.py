# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:34:31 2020

@author: pc
"""

import numpy as np

def bark2freq(x):
    return 600*np.sinh(x/6)

def freq2bark(x):
    return 6*np.arcsinh(x/600)



