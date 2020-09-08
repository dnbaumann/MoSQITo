# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4  2020
@author: pc
"""

import sys
sys.path.append(r"C:/Users/pc/Documents/Salomé/MoSQITo_oo")
#sys.path.append('../../..')

#standard library imports
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

#local applications imports
from mosqito.functions.oct3filter.calc_third_octave_levels import calc_third_octave_levels
from mosqito.functions.oct3filter.oct3spec import oct3spec
from mosqito.functions.loudness_zwicker.loudness_zwicker_shared import calc_main_loudness
from mosqito.functions.loudness_zwicker.loudness_zwicker_nonlinear_decay import calc_nl_loudness
from mosqito.functions.loudness_zwicker.loudness_zwicker_shared import calc_slopes
from mosqito.functions.loudness_zwicker.loudness_zwicker_temporal_weighting import loudness_zwicker_temporal_weighting
from mosqito.functions.loudness_zwicker.loudness_zwicker_stationary import loudness_zwicker_stationary
from mosqito.functions.loudness_zwicker.loudness_zwicker_time import loudness_zwicker_time
from mosqito.functions.conversion_bark2frequency import (bark2freq, freq2bark)

class Audio_signal:
    """ Audio signal to be analyzed """
    

    def __init__(self, signal_type, file = '', spec = [0,0]):
        
        """Parameters
        ---------------
        file : string path to the signal file
        type : stationary / time_varying 
        spec : third octave band spectrum
       
        """
      
        self.file = file
        self.signal_type = signal_type
        self.signal = np.array((1,1),dtype=float)
        self.fs = int()         
        self.spec = spec
        self.freq = np.array([
            25,
            31.5,
            40,
            50,
            63,
            80,
            100,
            125,
            160,
            200,
            250,
            315,
            400,
            500,
            630,
            800,
            1000,
            1250,
            1600,
            2000,
            2500,
            3150,
            4000,
            5000,
            6300,
            8000,
            10000,
            12500,  ] )
        self.N = float()
        self.N_specific = np.ndarray((1))
        
        if signal_type !='stationary' and signal_type!='time_varying':
            raise ValueError("ERROR: signal_type should be either 'stationary' or 'time_varying'.")
                
        
    def load_wav(self):
        """ Load .wav signal and affects its sampling frequency and time signal values 
        
        Parameters
        ----------
        calib : float calibration factor for the signal to be in [pa]

        Outputs
        -------
        signal : time signal values
        fs : sampling frequency        
        """
        
        self.fs, self.signal = wavfile.read(self.file)
        calib= 2 * 2**0.5
        if isinstance(self.signal[0], np.int16):
             self.signal = calib * self.signal / (2 ** 15 - 1)
        elif isinstance(self.signal[0], np.int32):
             self.signal = calib * self.signal / (2 ** 31 - 1)
             
             
      
    def plot_time(self):
        """ Time signal wave plotting """
        
        time = np.linspace(0, len(self.signal)/self.fs, num=len(self.signal))    
        plt.figure(1)
        plt.title("Signal Wave")
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.ylim( min(self.signal), max(self.signal))
        plt.plot(time, self.signal)
        plt.show()

  
        
    
    def comp_third_oct(self):       
        """ Third-octave band spectrum calculation, with the corresponding 
            bands center frequencies
        
           Outputs
          --------
          spec : numpy.ndarray
                 Third octave band spectrum of signal sig [dB re.2e-5 Pa]
          freq : numpy.ndarray
                 Corresponding third octave bands center frequencies                   
        """                                  
    #test signal
    
        if self.signal_type == 'stationary':
            self.spec, self.freq = oct3spec(self.signal, self.fs)
        elif self.signal_type == 'time_varying':
            self.spec = calc_third_octave_levels(self.signal,self.fs)
                   
            
        
        # Idea : third-oct calculation of a time-varying signal using a stationary similar  method
        # dec_factor = int(self.fs / 2000)
        #     self.spec, self.freq = oct3spec(self.signal, self.fs, 25, 12500, self.type, dec_factor=24)
                  
        np.squeeze(self.spec)
        
        #TO DO : third-oct values calculation from a fine band spectrum
        
        
    def plot_freq(self):
        """Amplitudes related to frequencies plotting"""
        
        plt.step(self.freq, self.spec)
        plt.xscale('log')
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude, [dB re. 2.10^-5 Pa]")
        plt.show()
        
        
        
    def comp_loudness(self, field_type = 'free'):
        """  Acoustic loudness calculation according to Zwicker method for
        stationary and time-varying signals."""
        
        if self.signal_type == 'stationary':
            self.N, self.N_specific = loudness_zwicker_stationary(self.spec, self.freq, field_type)
            print("Loudness:",str(self.N),"sones")
        elif self.signal_type == 'time_varying':
            self.N, self.N_specific = loudness_zwicker_time(self.spec, field_type)
            
        
   
    def plot_loudness(self):
        """ Specific band loudness plotting  """


        if self.signal_type == 'stationary':
            plt.figure(1)                
            fig, ax = plt.subplots(constrained_layout=True)
            x = np.linspace(0.1, 24, int(24 / 0.1))   
            x = x.astype(float)
            ax.plot(x, self.N_specific)
            ax.set_xlabel('Bark scale')
            ax.set_ylabel('Loudness [sones]')
            ax.set_title('Specific loudness')
            secax = ax.secondary_xaxis('top', functions=(bark2freq, freq2bark))
            plt.setp(secax.get_xticklabels(), rotation=60, ha="right")
            plt.setp(secax.set_xticks([0,100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500]))
            secax.set_xlabel('Frequency')
            plt.show()
        elif self.signal_type == 'time_varying':
            plt.figure(2)                
            fig, ax = plt.subplots(constrained_layout=True)
            time = np.linspace(0,0.002*(self.N.size - 1),self.N.size)            
            plt.plot(time, self.N)
            plt.xlabel("Time [s]")
            plt.ylabel("Loudness, [sone]")
            plt.title("Loudness over time")
            plt.show()



 
if __name__ == "__main__":
     audio=Audio_signal('stationary',r"C:\Users\pc\Documents\Salomé\MoSQITo_oo\mosqito\tests\data\ISO_532-1\Test signal 2 (250 Hz 80 dB).wav")    
     audio.load_wav()
     audio.plot_time()
     audio.comp_third_oct()
     audio.plot_freq()
     audio.comp_loudness()
     audio.plot_loudness()
     