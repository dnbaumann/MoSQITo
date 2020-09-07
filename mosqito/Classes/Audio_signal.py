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



class Audio_signal:
    """ Audio signal to be analyzed """
    

    def __init__(self, type, file = '', spec = [0,0]):
        
        """Parameters
        ---------------
        file : string path to the signal file
        type : stationary / time_varying 
        spec : third octave band spectrum
       
        """
      
        self.file = file
        self.type = type
        self.signal = np.array((1,1),dtype=float)
        self.fs = int()         
        self.spec = spec
        self.freq = np.array((1,1),dtype=int)
        self.N = float()
        self.N_specific = np.ndarray((1))
        
        
        
    def load_wav(self):
        """ Load .wav signal and output its sampling frequency and time signal values 
        
        Parameters
        ----------
        calib : float calibration factor for the signal to be in [pa]

        Outputs
        -------
        signal : time signal values
        fs : sampling frequency        
        """
        
        self.fs, self.signal = wavfile.read(self.file)
        calib=1
        if isinstance(self.signal[0], np.int16):
             self.signal = calib * self.signal / (2 ** 15 - 1)
        elif isinstance(self.signal[0], np.int32):
             self.signal = calib * self.signal / (2 ** 31 - 1)
             
             
      
    def signal_plot(self):
        """ Signal wave plotting """
        
        time = np.linspace(0, len(self.signal)/self.fs, num=len(self.signal))    
        plt.figure(1)
        plt.title("Signal Wave")
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.plot(time, self.signal)
        plt.show()

    
    
    def third_oct_filter(self):
        
        """Parameters 
          -----------        
           out_type : str
               determines the format of the output :
                - overall: overall rms value per third octave band
                - time: rms value per third octave versus time (temporal resolution = 0.5ms)
                - time_iso: squared and smoothed value per third octave versus time,
                  ISO 532-1 implementation (temporal resolution = 0.5ms)
              
           Outputs 
          --------
          spec : numpy.ndarray
                 Third octave band spectrum of signal sig [dB re.2e-5 Pa]
          freq : numpy.ndarray
                 Corresponding third octave bands center frequencies                   
        """                                  
    
        if self.type ==  'stationary':
            self.spec, self.freq = oct3spec(self.signal, self.fs, 25, 12500, self.type)
        elif self.type == 'time_varying':
            dec_factor = int(self.fs / 2000)
            self.spec, self.freq = oct3spec(self.signal, self.fs, 25, 12500, self.type, dec_factor=24)
        # elif out_type == 'time_iso':
        #     self.spec = calc_third_octave_levels(self.signal,self.fs)
        #     self.freq = np.array([
        #     25,
        #     31.5,
        #     40,
        #     50,
        #     63,
        #     80,
        #     100,
        #     125,
        #     160,
        #     200,
        #     250,
        #     315,
        #     400,
        #     500,
        #     630,
        #     800,
        #     1000,
        #     1250,
        #     1600,
        #     2000,
        #     2500,
        #     3150,
        #     4000,
        #     5000,
        #     6300,
        #     8000,
        #     10000,
        #     12500,  ] )
            
        np.squeeze(self.spec)
        
    def loudness_zwicker_stationary(self, field_type = 'free'):
        """  Acoustic loudness calculation according to Zwicker method for
        stationary signals.
        Normatice reference:
            ISO 532:1975 (method B)
            DIN 45631:1991
            ISO 532-1:2017 (method 1)
        The code is based on BASIC program published in "Program for
        calculating loudness according to DIN 45631 (ISO 532B)", E.Zwicker
        and H.Fastl, J.A.S.J (E) 12, 1 (1991). 
        Note that for reasons of normative continuity, as defined in the
        preceeding standards, the method is in accordance with 
        ISO 226:1987 equal loudness contours (instead of ISO 226:2003)

        Parameters
        ----------        
        third_axis : numpy.ndarray
                   Normalized center frequency of third octave bands [Hz]
        field_type : type of soundfield corresponding to spec_third
                    ("free" by default or "diffuse")
        

        Outputs
        -------
        N : Calculated loudness [sones]
        N_specific :  Specific loudness [sones/bark]
        bark_axis : numpy.ndarray
                    Corresponding bark axis
        """
        
        # Input parameters control and formating
        fr = [
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
            12500,
                    ]
        if field_type != "diffuse" and field_type != "free":
            raise ValueError("ERROR: field_type shall be either 'diffuse' or 'free'")
        if self.type != "stationary":
            raise ValueError("ERROR : signal_type shall be 'stationary'")
        
        else:
        # TODO: manage spectrum and third_axis longer than 28 (extract data
        # between 25 and 12.5 kHz)
            pass
        #    if (min(spec_third) < -60 or max(spec_third) > 120):
        # TODO: replace value below -60 by -60 and raise a warning
        #        raise ValueError(
        #            """ERROR: Third octave levels must be within interval
        #            [-60, 120] dB ref. 2e-5 Pa (for model validity) """
        #        )
    
    
        # Calculate main loudness
        Nm = calc_main_loudness(self.spec, field_type)
        #
        # Calculation of specific loudness pattern and integration of overall 
        # loudness by attaching slopes towards higher frequencies
        self.N, self.N_specific = calc_slopes(Nm)
        #
        # Bark axis
        bark_axis = np.linspace(0.1, 24, int(24 / 0.1))
        
        
        
    def loudness_zwicker_time(self, field_type = 'free'):
        """ Acoustic loudness calculation according to Zwicker method for
        time-varying signals
        Normatice reference:
            DIN 45631/A1:2010
            ISO 532-1:2017 (method 2)
        The code is based on C program source code published alongside
        with ISO 532-1 standard. 
        Note that for reasons of normative continuity, as defined in the
        preceeding standards, the method is in accordance with 
        ISO 226:1987 equal loudness contours (instead of ISO 226:2003)


        Outputs
        -------
        N : float
            Calculated loudness [sones]
        N_specific : numpy.ndarray
            Specific loudness [sones/bark]
        bark_axis : numpy.ndarray
            Corresponding bark axis
            """
        
        if self.type != "time_varying":
            raise ValueError("ERROR : signal_type shall be 'time_varying'")
            
        # Calculate core loudness
        num_sample_level = np.shape(self.spec)[1]
        core_loudness = np.zeros((21, num_sample_level))
        for i in np.arange(num_sample_level-1):
            core_loudness[:,i] = calc_main_loudness(self.spec[:,i],field_type)
        
        # Nonlinearity
        core_loudness = calc_nl_loudness(core_loudness)
        #
        # Calculation of specific loudness
        loudness = np.zeros(np.shape(core_loudness)[1])
        spec_loudness = np.zeros((240,np.shape(core_loudness)[1]))
        for i_time in np.arange(np.shape(core_loudness)[1]):
            loudness[i_time], spec_loudness[:,i_time] = calc_slopes(core_loudness[:,i_time])
            
        # temporal weigthing
        filt_loudness = loudness_zwicker_temporal_weighting(loudness)
            
        # Decimation from temporal resolution 0.5 ms to 2ms and return
        dec_factor = 4
        self.N = filt_loudness[::dec_factor]
        self.N_spec = spec_loudness[:,::dec_factor]
        bark_axis = np.linspace(0.1, 24, int(24 / 0.1))
    

