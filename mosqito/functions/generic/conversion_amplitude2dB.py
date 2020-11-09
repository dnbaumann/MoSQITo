# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 18:02:06 2020

@author: pc
"""
import numpy as np


def db2amp(db):
    """ Linearisation of a SPL level in dB 
        reference = 2e-05                  """
    return np.power(10,0.05*db)*0.00002

def amp2db(amp):
    """ Conversion of an amplitude value into dB 
        reference = 2e-05                       """
    
    return 20*np.log10(amp/0.00002) 