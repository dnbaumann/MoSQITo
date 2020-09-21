# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 13:54:40 2020

@author: pc
"""

import pyuff


def uff_loader(file):
    """ Extract the signal and its time axis from a .uff file
    
    Parameter:
    ----------
    file : str()
           path to the signal file
           
    Outputs:
    --------
    signal: np.array of float
            signal values along time
    fs: sampling frequency
    
    """
# loading the uff file content
    uff_file = pyuff.UFF(file)
    data = uff_file.read_sets()
    data.keys()
    
    
#extracting the signal values
    signal = data['data']
    # time_axis = data['x']
    
# calculating the sampling frequency
    fs = int(1/data['abscissa_inc'])
    
    return fs, signal