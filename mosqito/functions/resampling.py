# -*- coding: utf-8 -*-
"""
Created on 18/09/2020
"""

from scipy import signal
import numpy as np


def signal_resample (audio, fs):
    """ Changes signal sampling frequency to 48 kHZ"""
    
    
    duration = int(len(audio)/fs)
    audio = signal.resample(audio, 48000*duration)
    audio = audio.astype(np.int16)
    fs = 48000
    
    return fs, audio

