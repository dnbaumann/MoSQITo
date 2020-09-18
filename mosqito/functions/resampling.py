# -*- coding: utf-8 -*-
"""
Created on 18/09/2020
"""

from scipy.io import wavfile
from scipy import signal


def signal_resample (audio):
    """ Change a signal sampling frequency to 48 kHZ"""


    audio = signal.resample(audio, 48000)
    fs = 48000
    
    return fs, audio

