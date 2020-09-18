# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4  2020
@author: pc
"""

import sys
sys.path.append('../..')

#standard library imports
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

#local applications imports
from mosqito.functions.oct3filter.calc_third_octave_levels import calc_third_octave_levels
from mosqito.functions.oct3filter.oct3spec import oct3spec
from mosqito.functions.loudness_zwicker.loudness_zwicker_stationary import loudness_zwicker_stationary
from mosqito.functions.loudness_zwicker.loudness_zwicker_time import loudness_zwicker_time
from mosqito.functions.conversion_bark2frequency import (bark2freq, freq2bark)
from mosqito.functions.third_oct_2_dBA import A_weighting
from mosqito.functions.resampling import signal_resample
# from mosqito.functions.sharpness.sharpness_aures import calc_sharpness_aures
# from mosqito.functions.sharpness.sharpness_din import calc_sharpness_din
# from mosqito.functions.sharpness.sharpness_bismarck import calc_sharpness_bismarck


class Audio_signal:
    """ Audio signal to be analyzed """
    

    def __init__(self):
        
        self.file = str()
        self.is_stationary = bool()
        self.signal = np.array((1,1),dtype=float)
        self.fs = int()         
        self.spec_third = np.ndarray((1,1))
        self.sepc_dBA = np.ndarray((1,1))
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
            12500  ] )
        self.N = float()
        self.N_specific = np.ndarray((1))
        self.S_aures = np.ndarray((1))
        self.S_din = np.ndarray((1))
        self.S_bismarck = np.ndarray((1))
        self.R = float()
        self.R_specific = np.ndarray((1))
        
        
    def load_wav(self, is_stationary, file, calib ):
        """ Load .wav signal and affects its sampling frequency and time signal values 
        
        Parameters
        ----------
        file : string path to the signal file
        is_stationary : boolean 
                 True if the signal is stationary, False if it is time-varying
        calib : float calibration factor for the signal to be in [pa]

        Outputs
        -------
        signal : time signal values
        fs : sampling frequency        
        """
        
        self.is_stationary = is_stationary
        self.file = file
        self.fs, self.signal = wavfile.read(self.file)
        if self.fs != 48000:
            self.fs, self.signal = signal_resample(self.signal)
        if isinstance(self.signal[0], np.int16):
             self.signal = calib * self.signal / (2 ** 15 - 1)
        elif isinstance(self.signal[0], np.int32):
             self.signal = calib * self.signal / (2 ** 31 - 1)
             
                   
    def plot_time(self):
        """ Time signal wave plotting """
        
        time = np.linspace(0, len(self.signal)/self.fs, num=len(self.signal))    
        plt.figure()
        plt.title("Signal Wave")
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.ylim( min(self.signal), max(self.signal))
        plt.plot(time, self.signal)
        plt.show()

    def set_third_oct(self, spec_third, third_axis):
        """ Load a third-octave band spectrum and its corresponding 
            bands center frequencies. 
        
        Parameters
        ----------
        spec_third : numpy.ndarray
                  A third octave band spectrum [dB ref. 2e-5 Pa]
        third_axis : numpy.ndarray
                  Normalized center frequency of third octave bands [Hz]
         
        """
        self.is_stationary = True
        if len(spec_third) != 28:
            raise ValueError("ERROR: spectrum must contains 28 third octave bands values")
        elif (len(third_axis) == 28 and np.all(third_axis != self.freq)) or len(third_axis) < 28:
            raise ValueError("""ERROR: third_axis does not contains 1/3 oct between 25 and 
            12.5 kHz. Check the input parameters""")
        self.spec_third = spec_third
        
    
    def comp_third_oct(self):       
        """ Third-octave band spectrum calculation, with the corresponding 
            bands center frequencies
        
           Outputs
          --------
          spec_third : numpy.ndarray
                 Third octave band spectrum of signal sig [dB re.2e-5 Pa]
          freq : numpy.ndarray
                 Corresponding third octave bands center frequencies                   
        """                                  
    #test signal
    
        if self.is_stationary == True:
            self.spec_third, self.freq = oct3spec(self.signal, self.fs)
        elif self.is_stationary == False:
            self.spec_third = calc_third_octave_levels(self.signal,self.fs)
                   
            
        
        # Idea : third-oct calculation of a time-varying signal using a stationary similar  method
        # dec_factor = int(self.fs / 2000)
        #     self.spec, self.freq = oct3spec(self.signal, self.fs, 25, 12500, self.type, dec_factor=24)
                  
        np.squeeze(self.spec_third)
        
        #TO DO : third-oct values calculation from a fine band spectrum
        
        
    def plot_freq(self,unit):
        """Amplitudes related to frequencies plotting
        
        Parameter:
        --------------
        unit : str()
             'dB' or 'dBA'
                               
        """
        plt.figure()
        
        if unit =='dB':
            plt.step(self.freq, self.spec_third)
            plt.title('Third octave spectrum')
            plt.xscale('log')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude, [dB re. 2.10^-5 Pa]")
            plt.show()
        elif unit =='dBA':
            self.spec_dBA = A_weighting(self.spec_third)
            plt.step(self.freq, self.spec_dBA)
            plt.xscale('log')
            plt.title('A-weighted third-octave spectrum')
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [dBA]')
            plt.show()
        

       
    
    def comp_loudness(self, field_type = 'free'):
        """  Acoustic loudness calculation according to Zwicker method for
        stationary and time-varying signals."""
        
        if self.is_stationary == True:
            self.N, self.N_specific = loudness_zwicker_stationary(self.spec_third, self.freq, field_type)
            print("Loudness:",str(self.N),"sones")
        elif self.is_stationary == False:
            self.N, self.N_specific = loudness_zwicker_time(self.spec_third, field_type)
            
        
   
    def plot_loudness(self):
        """ Specific band loudness plotting  """


        if self.is_stationary == True:
            plt.figure()                
            fig, ax = plt.subplots(constrained_layout=True)
            x = np.linspace(0.1, 24, int(24 / 0.1))   
            x = x.astype(float)
            ax.plot(x, self.N_specific)
            ax.set_xlabel('Bark scale')
            ax.set_ylabel('Loudness [sones]')
            ax.set_title('Specific loudness')
            secax = ax.secondary_xaxis('top', functions=(bark2freq, freq2bark))
            plt.setp(secax.get_xticklabels(), rotation=60, ha="right")
            secax.set_xticks(np.array([25,50,100,200,400,800,1600,3150,6300,12500]))
            secax.set_xlabel('Frequency')
            plt.show()
        elif self.is_stationary == False:
            plt.figure()                
            fig, ax = plt.subplots(constrained_layout=True)
            time = np.linspace(0,0.002*(self.N.size - 1),self.N.size)            
            plt.plot(time, self.N)
            plt.xlabel("Time [s]")
            plt.ylabel("Loudness, [sone]")
            plt.title("Loudness over time")
            plt.show()
            
            
            
### Work in progress : sharpness implementation

    # def comp_sharpness(self):
    #     """ Acoustic sharpness calculation according to
    #         different methods
            
            
    #         Output
    #         ------
    #         S : float
    #             sharpness value
                       
    #         """
         
    #     self.S_aures = calc_sharpness_aures(self.N, self.N_specific, self.is_stationary )              
    #     self.S_din = calc_sharpness_din(self.N, self.N_specific, self.is_stationary)
    #     self.S_bismarck = calc_sharpness_bismarck(self.N, self.N_specific, self.is_stationary)        
        
    # def plot_sharpness(self):
    #     """ Sharpness plotting """

    #     if self.is_stationary == False:
    #         plt.figure()                
    #         time = np.linspace(0,0.002*(self.N.size - 1),self.N.size)            
    #         plt.plot(time, self.S_aures, label='Aures', color='blue')
    #         plt.plot(time, self.S_din, label='DIN', color='red')
    #         plt.plot(time, self.S_bismarck, label='Von Bismarck', color='orange')
    #         plt.xlabel("Time [s]")
    #         plt.ylabel("Sharpness, [acum]")
    #         plt.title("Sharpness over time")
    #         plt.legend()
    #         plt.show()


 
if __name__ == "__main__":
# ##test : loudness calculation from a third_octave band spectrum (steady signal)
#       test_signal_1 = np.array([
#     -60, -60, 78, 79, 89, 72, 80, 89, 75, 87, 85, 79, 86, 80, 71, 70, 72, 71,
#     72, 74, 69, 65, 67, 77, 68, 58, 45, 30])
#       fr = [ 25, 31.5, 40, 50,63, 80,  100, 125,  160,200, 250, 315, 400, 500,
#             630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
#             8000, 10000, 12500,  ] 
#       audio = Audio_signal()         
#       audio.set_third_oct(test_signal_1, fr)  
#       audio.plot_freq('dBA')
#       audio.comp_loudness()
#       audio.plot_loudness()

# # test : loudness calculation from a .wav file (steady signal)
#     audio = Audio_signal()  
#     audio.load_wav(True, r"C:\Users\pc\Documents\Salomé\MoSQITo_oo\mosqito\tests\data\ISO_532-1\Test signal 2 (250 Hz 80 dB).wav", calib = 2 * 2**0.5)
#     audio.plot_time()
#     audio.comp_third_oct()
#     audio.plot_freq('dBA')
#     audio.comp_loudness()
#     audio.plot_loudness()
      
     
# # #test : loudness calculation from a .wav file (time_varying signal)   
#         audio = Audio_signal()   
#         audio.load_wav(False,"mosqito\tests\data\ISO_532-1\Annex B.5\Test signal 17 (machine gun).wav", calib = 2 * 2**0.5)
#         #audio.plot_time()
#         audio.comp_third_oct()
#         audio.comp_loudness()
#         audio.plot_loudness()
  