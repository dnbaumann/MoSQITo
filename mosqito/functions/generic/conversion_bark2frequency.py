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
   
    xp = np.arange(0,25,0.5)        
    
    yp = np.array([   0,    50,   100,   150,   200,   250,   300,   350,   400,
                    450,   510,   570,   630,   700,   770,   840,   920,  1000,
                   1080,  1170,  1270,  1370,  1480,  1600,  1720,  1850,  2000,
                   2150,  2320,  2500,  2700,  2900,  3150,  3400,  3700,  4000,
                   4400,  4800,  5300,  5800,  6400,  7000,  7700,  8500,  9500,
                  10500, 12000, 13500, 15500, 20000])
    
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
    
    xp = np.array([   0,    50,   100,   150,   200,   250,   300,   350,   400,
                    450,   510,   570,   630,   700,   770,   840,   920,  1000,
                   1080,  1170,  1270,  1370,  1480,  1600,  1720,  1850,  2000,
                   2150,  2320,  2500,  2700,  2900,  3150,  3400,  3700,  4000,
                   4400,  4800,  5300,  5800,  6400,  7000,  7700,  8500,  9500,
                  10500, 12000, 13500, 15500, 20000])
    
    yp = np.arange(0,25,0.5)    
    
    bark_axis = np.interp(freq_axis,xp,yp)
    
    return bark_axis


