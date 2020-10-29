# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4  2020
@author: wantysal
"""

import sys
sys.path.append('../..')

#standard library imports
import numpy as np
from numpy.fft import fft
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt


#local applications imports
from mosqito.functions.generic.uff_loader import uff_loader
from mosqito.functions.generic.spectrum_2_dBA import A_weighting
from mosqito.functions.generic.resampling import signal_resample
from mosqito.functions.generic.conversion_bark2frequency import (bark2freq, freq2bark)
from mosqito.functions.oct3filter.calc_third_octave_levels import calc_third_octave_levels
from mosqito.functions.oct3filter.oct3spec import oct3spec
from mosqito.functions.loudness_zwicker.loudness_zwicker_stationary import loudness_zwicker_stationary
from mosqito.functions.loudness_zwicker.loudness_zwicker_time import loudness_zwicker_time
from mosqito.functions.roughness.roughness import calc_roughness
from mosqito.functions.sharpness.sharpness_aures import calc_sharpness_aures
from mosqito.functions.sharpness.sharpness_din import calc_sharpness_din
from mosqito.functions.sharpness.sharpness_bismarck import calc_sharpness_bismarck

# set matplotlib font
plt.rcParams['font.sans-serif'] = "MC Gothic"
plt.rcParams['font.family'] = "sans-serif"


class Audio_signal:
    """ Audio signal loading and analysis: from .wav or .uff files, 
        compute loudness, sharpness and roughness values. """
    

    def __init__(self):
        """Constructor of the class."""
        
        self.file = str()
        self.is_stationary = bool()
        self.signal = np.array((1,1),dtype=float)
        self.fs = int()  
        self.time_axis = np.ndarray((1))
        self.spec_dB = np.ndarray((1,1))
        self.spec_dBA = np.ndarray((1,1))
        self.period = np.ndarray((1,1))
        self.period_freq = np.ndarray((1))
        self.spec_third_dB = np.ndarray((1,1))
        self.spec_third_dBA = np.ndarray((1,1))        
        self.freq = np.array([
            25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
            1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000, 12500  ])
        self.level_dB = np.ndarray((1,1))
        self.level_dBA = np.ndarray((1,1))
        self.N = float()
        self.N_specific = np.ndarray((1))
        self.S_aures = np.ndarray((1))
        self.S_din = np.ndarray((1))
        self.S_bismarck = np.ndarray((1))
        self.R = float()
        
        
    def load_wav(self, is_stationary, file, calib=1 ):
        """ Load .wav signal and affects its sampling frequency and time signal values 
        
        Parameters
        ----------
        file : string
              string path to the signal file
        is_stationary : boolean
                      True if the signal is stationary, False if it is time-varying
        calib : float
               calibration factor for the signal to be in [pa]

        Outputs
        -------
        signal : numpy.array
               time signal values
        fs : integer
            sampling frequency        
        """
        
        self.is_stationary = is_stationary
        self.file = file
        self.fs, self.signal = wavfile.read(self.file) 
    
        if isinstance(self.signal[0], np.int16):
             self.signal = calib * self.signal / (2 ** 15 - 1)
        elif isinstance(self.signal[0], np.int32):
             self.signal = calib * self.signal / (2 ** 31 - 1)  
             
        if self.fs != 48000:
            self.fs, self.signal = signal_resample(self.signal,self.fs)
            print("Signal resampled to 48 kHz to allow calculation.")
        self.time_axis = np.linspace(0, len(self.signal)/self.fs, num=len(self.signal))
    
    def load_uff(self, is_stationary, file):
        """ Load .uff signal and affects its sampling frequency and time signal values 
        
        Parameters
        ----------
        file : string
             string path to the signal file
        is_stationary : boolean 
                 True if the signal is stationary, False if it is time-varying

        Outputs
        -------
        signal : numpy.array
               time signal values
        fs : integer
            sampling frequency          
        """ 
        
        self.is_stationary = is_stationary
        self.file = file
        self.fs, self.signal = uff_loader(self.file)   
        if self.fs != 48000:
            self.signal = signal.resample(self.signal, 48000*int(self.signal.size/self.fs))
            self.fs = 48000
            print("Signal resampled to 48 kHz to allow calculation.")
        self.time_axis = np.linspace(0, len(self.signal)/self.fs, num=len(self.signal))
    
                   
    def plot_signal(self):
        """ Time signal wave plot """
        
        plt.figure(num='Signal wave')
        time = np.linspace(0, len(self.signal)/self.fs, num=len(self.signal))    
        plt.title("Signal Wave",fontsize=16)
        plt.ylabel("Amplitude")
        plt.xlabel("Time")
        plt.ylim( min(self.signal), max(self.signal))
        plt.plot(time, self.signal,color='#69C3C5')
        plt.show()
    
    def plot_periodogram(self,start,stop):
        """ Signal periodogram calculation and plot in the given time frame 
        
        Parameters
        ----------
        start : integer
               when to start computing spectrum in s
        stop : integer
              when to stop computing spectrum in s 
                
        """
        
        self.period_freq, self.period = signal.periodogram(self.signal[int(start*self.fs):int(stop*self.fs)],self.fs)                      
        plt.figure(figsize=(12,4),num='Periodogram')           
        plt.title('Audio signal periodogram',fontsize=16)
        plt.plot(self.period_freq,self.period,color='#69C3C5')  
        plt.axis([0,10000,self.period.min(),self.period.max()+0.05])
        plt.ylabel('Amplitude [dB]')
        plt.xlabel('Frequency [Hz]')        
        plt.grid()  
        
    
    def comp_spectrum(self,start,stop,average,window):
        """ Signal spectrum calculation in the given time frame 
            with the chosen window
        
        Parameters
        ----------
        start : integer
               when to start computing spectrum in s
        stop : integer
              when to stop computing spectrum in s 
        average : boolean
                  if true, the spectrum is calculated from periodogram
        window : string
                'Blackman' or 'Hamming'   
                
        """
        
        # signal extracted within the given time frame
        sig = self.signal[int(start*self.fs):int(stop*self.fs)]
        N = sig.size
        
        # spectrum calculation
        if average == True:
            self.period_freq, self.period = signal.periodogram(self.signal[int(start*self.fs):int(stop*self.fs)],self.fs) 
            self.spec_dB = 10*np.log10(self.period/np.power(0.00002,2))
        if average == False:        
            if window == 'blackman':
                fourier = np.absolute(fft(sig*np.blackman(N))*(1/int(N/2)))[0:int(N/2)]
            elif window == 'hamming':
                fourier = np.absolute(fft(sig*np.hamming(N))*(1/int(N/2)))[0:int(N/2)]            
            self.spec_dB = 20 * np.log10(fourier/0.00002)    
        
        # A weighting 
        self.spec_dBA = A_weighting(self.spec_dB,self.fs)
    
    def plot_spectrum(self,start,stop,average,window,unit):
        """ Signal spectrum calculation and plot in the given time frame 
            with the chosen window and unit
        
        Parameters
        ----------
        start : integer
               when to start computing spectrum in s
        stop : integer
              when to stop computing spectrum in s 
        average : boolean
                  if true, the spectrum is calculated from periodogram
        window : string
                'Blackman' or 'Hamming'   
        unit : string
              'dB' or 'dBA'
                
        """
        
        # signal extracted within the given time frame
        sig = self.signal[int(start*self.fs):int(stop*self.fs)]
        N = sig.size
        
        # spectrum calculation
        if average == True:
            self.period_freq, self.period = signal.periodogram(self.signal[int(start*self.fs):int(stop*self.fs)],self.fs) 
            self.spec_dB = 10*np.log10(self.period/np.power(0.00002,2))
        if average == False:        
            if window == 'blackman':
                fourier = np.absolute(fft(sig*np.blackman(N))*(1/int(N/2)))[0:int(N/2)]
            elif window == 'hamming':
                fourier = np.absolute(fft(sig*np.hamming(N))*(1/int(N/2)))[0:int(N/2)]            
            self.spec_dB = 20 * np.log10(fourier/0.00002)    
        
        # A weighting 
        self.spec_dBA = A_weighting(self.spec_dB,self.fs)
              
        # corresponding frequency axis        
        freq = np.linspace(0,int(self.fs/2),self.spec_dB.size)        
        
        # plot in the chosen dB unit
        plt.figure(figsize=(12,4),num='Spectrum') 
        if unit == 'dB':            
            plt.title('Audio signal spectrum',fontsize=16)
            plt.plot(freq,self.spec_dB,color='#69C3C5')  
            plt.axis([0,10000,-40,self.spec_dB.max()+10])
            plt.ylabel('Amplitude [dB]')
            plt.xlabel('Frequency [Hz]')
        if unit == 'dBA':            
            plt.figure(figsize=(12,4),num='Spectrum') 
            plt.title('Audio signal spectrum',fontsize=16)
            plt.plot(freq,self.spec_dBA,color='#69C3C5') 
            plt.axis([0,10000,-40,self.spec_dBA.max()+10])
            plt.ylabel('Amplitude [dBA]')
            plt.xlabel('Frequency [Hz]')
        plt.grid()                   
                    
    
    
    def set_third_oct(self, spec_third, third_axis):
        """ Load a steady signal third-octave band spectrum and its corresponding 
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
        self.spec_third_dB = spec_third
    
    
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
    
        if self.is_stationary == True:
            self.spec_third_dB, self.freq = oct3spec(self.signal, self.fs)
        elif self.is_stationary == False:            
            self.spec_third_dB = calc_third_octave_levels(self.signal,self.fs)
        np.squeeze(self.spec_third_dB)           
            
        
    # Idea : third-oct calculation of a time-varying signal using 
    # a stationary similar  method dec_factor = int(self.fs / 2000)
    # self.spec, self.freq = oct3spec(self.signal, self.fs, 25, 12500, self.type, dec_factor=24)
         
    #TO DO : third-oct values calculation from a fine band spectrum
        
        
    def plot_freq(self,unit):
        """Amplitude related to frequency plot in the chosen unit.
        
        Parameter:
        --------------
        unit : string
             'dB' or 'dBA' to apply A-weighting
                               
        """
        plt.figure(num='Third-octave spectrum')
        
        if unit =='dB':
            plt.step(self.freq, self.spec_third_dB,color='#69C3C5')
            plt.title('Third octave spectrum',fontsize=16)
            plt.xscale('log')
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Amplitude, [dB re. 2.10^-5 Pa]")
        elif unit =='dBA':
            self.spec_third_dBA = A_weighting(self.spec_third,self.fs)
            plt.step(self.freq, self.spec_third_dBA,color='#69C3C5')
            plt.xscale('log')
            plt.title('A-weighted third-octave spectrum',fontsize=16)
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Amplitude [dBA]')
        plt.show()
     
    
    def comp_level(self, unit):
        """ Overall Sound Pressure Level calculation in the chosen unit 
        
        Parameter:
        ----------
        unit : string
             'dB' or 'dBA' to apply A-weighting
            
        """
        # Third octave spectrum calculation
        audio.comp_third()
        
        
        # Overall level calculation for a steady signal
        if self.is_stationary == True:
            if unit =='dB':
                self.level_dB = 10 * np.log10(sum(np.power(10,self.spec_third_dB/10)))
                print('Overall level: '+ str(self.level_dB) +' dB')
            elif unit =='dBA':
                self.spec_third_dBA = A_weighting(self.spec_third_dB,self.fs)
                self.level_dBA = 10 * np.log10(sum(np.power(10,self.spec_third_dBA/10)))
                print('Overall level: '+ str(self.level_dBA) +' dBA')
        
        # Overall level calculation for a time-varying signal
        elif self.is_stationary == False:
            time_axis = np.linspace(0, self.signal.size/self.fs, self.spec_third_dB.shape[1])
            if unit =='dB':
                self.level_dB = 10 * np.log10(sum(np.power(10,self.spec_third_dB/10)))
                plt.plot(time_axis, self.level_dB,color='#69C3C5',linewidth='2')
                plt.title('Overall Sound Pressure Level in dB',fontsize=16)
                plt.xlabel("Time [s]")
                plt.ylabel(" Sound Pressure Level [dB re. 2.10^-5 Pa]")
            elif unit =='dBA':
                self.spec_third_dBA = A_weighting(self.spec_third_dB, self.fs)                
                self.level_dBA = 10 * np.log10(sum(np.power(10,self.spec_third_dBA/10)))
                plt.step(time_axis, self.level_dB,color='#69C3C5',linewidth='2')
                plt.title('Overall Sound Pressure Level in dBA',fontsize=16)
                plt.xlabel("Time [s]")
                plt.ylabel("Sound Pressure Level, [dBA]")
     
        
     
        
    
    def comp_loudness(self, field_type = 'free'):
        """  Acoustic loudness calculation according to Zwicker method for
        stationary and time-varying signals."""
        
        if self.is_stationary == True:
            self.N, self.N_specific = loudness_zwicker_stationary(self.spec_third_dB, self.freq, field_type)
            print("Loudness:",str(self.N),"Sones")
        elif self.is_stationary == False: 
            self.N, self.N_specific = loudness_zwicker_time(self.spec_third_dB, field_type)
            
        
   
    def plot_loudness(self):
        """ Specific loudness plot within the 24 critical bands """

        plt.figure(num='Loudness')
        if self.is_stationary == True:                            
            fig, ax = plt.subplots(constrained_layout=True)
            x = np.linspace(0.1, 24, int(24 / 0.1))   
            x = x.astype(float)
            ax.plot(x, self.N_specific,color='#69C3C5')
            ax.set_xlabel('Bark scale')
            ax.set_ylabel('Loudness [sones]')
            ax.set_title('Specific loudness',fontsize=16)
            secax = ax.secondary_xaxis('top', functions=(bark2freq, freq2bark))
            plt.setp(secax.get_xticklabels(), rotation=60, ha="right")
            secax.set_xticks(np.array([25,50,100,200,400,800,1600,3150,6300,12500]))
            secax.set_xlabel('Frequency')        
        elif self.is_stationary == False:               
            fig, ax = plt.subplots(constrained_layout=True)
            plt.plot(self.time_axis, self.N,color='#69C3C5')
            plt.xlabel("Time [s]")
            plt.ylabel("Loudness, [sone]")
            plt.title("Loudness over time",fontsize=16)
        plt.show()
            
     
    def comp_roughness(self, overlap):
        """  Acoustic roughness calculation according to the Daniel and Weber method. 
        
        Parameter:
        ----------
        overlap: float
                overlapping coefficient for the time windows calculation
        """
                
        self.R, self.R_specific = calc_roughness(self.signal,self.fs, overlap)
        
        
   
    def plot_roughness(self):
        """ Roughness over time plot.  """
               
        plt.figure(num='Roughness')  
        time_axis = np.linspace(0,len(self.signal)/self.fs,self.R.size)       
        plt.plot(time_axis, self.R,color='#69C3C5')
        plt.ylim( 0, max(self.R)+ max(self.R)*0.1)
        plt.xlabel("Time [s]")
        plt.ylabel("Roughness [Asper]")
        plt.title("Roughness over time",fontsize=16)
        plt.show()
            
            
# ### Work in progress : sharpness implementation

    def comp_sharpness(self):
        """ Acoustic sharpness calculation according to
            different methods
            
            
            Output
            ------
            S : float
                sharpness value
                       
            """
         
        self.S_aures = calc_sharpness_aures(self.N, self.N_specific, self.is_stationary )              
        self.S_din = calc_sharpness_din(self.N, self.N_specific, self.is_stationary)
        self.S_bismarck = calc_sharpness_bismarck(self.N, self.N_specific, self.is_stationary)        
        
    def plot_sharpness(self):
        """ Sharpness plotting """

        if self.is_stationary == False:
            plt.figure(num='Roughness')                
            time = np.linspace(0,0.002*(self.N.size - 1),self.N.size)            
            plt.plot(time, self.S_aures, label='Aures', color='#0069A1')
            plt.plot(time, self.S_din, label='DIN', color='#69C3C5')
            plt.plot(time, self.S_bismarck, label='Von Bismarck', color='#E69F00')
            plt.xlabel("Time [s]")
            plt.ylabel("Sharpness, [acum]")
            plt.title("Sharpness over time",fontsize=16)
            plt.legend()
            plt.grid()



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

# # # test : loudness calculation from a .wav file (steady signal)
    
    # audio = Audio_signal()
    # audio.load_wav(True, file,calib=1)
    # audio.plot_signal()
    # audio.comp_third_oct()
    # #audio.plot_freq('dBA')
    # audio.comp_loudness()
    # audio.plot_loudness()
      
     
# # #test : loudness calculation from a .wav file (time_varying signal)   
# audio = Audio_signal()   
# audio.load_wav(False, r"example",calib = 1)
# # audio.plot_time()
# audio.comp_third_oct()
# audio.comp_loudness()
# audio.plot_loudness()
# audio.comp_sharpness()
# audio.plot_sharpness()

