# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 12:26:37 2020

@author: pc
"""

import numpy as np

class Audio_signal:
    """  """

    def __init__(self, file):
        self.file = file
        self.signal = np.array((1,1),dtype=float)
        self.fs = int() 
    
if __name__ == "__main__":
    
    audio=Audio_signal('test')