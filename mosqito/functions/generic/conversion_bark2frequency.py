# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 11:34:31 2020

@author: wantysal
"""

import numpy as np


def bark2freq(bark_axis):
    """ Frequency conversion from Bark to Hertz
    
    See E. Zwicker, H. Fastl: Psychoacoustics. Springer,Berlin, Heidelberg, 1990. 
    The coefficients are linearly interpolated from the values given in table 6.1.

    Parameter
    ---------
    bark_axis : numpy.array
                Bark frequencies to be converted
                
    Output
    ------
   freq_axis : numpy.array
               frequencies converted in Hertz
    
    """
   
    xp = np.arange(0,25,1)        
    
    yp = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720,
          2000, 2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
    
    freq_axis = np.interp(bark_axis,xp,yp)
        
    return freq_axis
    
def freq2bark(freq_axis):    
    """ Frequency conversion from Hertz to Bark
    
    See E. Zwicker, H. Fastl: Psychoacoustics. Springer,Berlin, Heidelberg, 1990. 
    The coefficients are linearly interpolated from the values given in table 6.1.

    Parameter
    ---------
    freq_axis : numpy.array
                Hertz frequencies to be converted
                
    Output
    ------
   bark_axis : numpy.array
              frequencies converted in Bark
    
    """
    
    xp = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 1270, 1480, 1720, 2000,
          2320, 2700, 3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500]
    
    yp = np.arange(0,25,1)
    
    bark_axis = np.interp(freq_axis,xp,yp)
    
    return bark_axis


