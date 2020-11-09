# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:11:27 2020

@author: pc
"""

import numpy as np


def a0tab(bark_axis):
    
    xp = [0,10,12,13,14,15,16,16.5,17,18,18.5,19,20,21,21.5,22,22.5,23,23.5,24,25,26]
    yp = [0,0,1.15,2.31,3.85,5.62,6.92,7.38,6.92,4.23,2.31,0,-1.43,-2.59,-3.57,-5.19,
          -7.41,-11.3,-20,-40,-130,-999]
    
    a0tab = np.interp(bark_axis,xp,yp)
    
    return a0tab