# -*- coding: utf-8 -*-
"""
Created on 18/09/2020
"""

from scipy import signal


def signal_resample (audio, fs):
    """ Changes signal sampling frequency to 48 kHZ 
    
    Parameters
    ----------
    audio: numpy.array
           signal values along time
    fs: integer
        original sampling frequency
    
    Outputs
    -------
    new_fs:numpy.array
            corrected sampling frequency = 48kHz        
    new_audio: numpy.array
                resampled signal 
        
    """
    
    
    duration = int(len(audio)/fs)
    new_audio = signal.resample(audio, 48000*duration)
    new_fs = 48000
    
    return new_fs, new_audio

