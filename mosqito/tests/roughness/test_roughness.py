# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 13:41:37 2020

@author: wantysal
"""

import sys
sys.path.append('../../..')

#Standard imports
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Local application imports
from mosqito.Classes.Audio_signal import Audio_signal
from mosqito.tests.roughness.data.signal_generator import signal_generation

@pytest.mark.roughness  # to skip or run only roughness tests
def test_roughness():
    """Test function for the roughness calculation of a audio signal

    Test function for the Audio_signal class "comp_roughness" method with signal array 
    as input. The input signals are chosen according to the article "Psychoacoustical 
    roughness: implementation of an optimized model" by Daniel and Weber in 1997.
    The figure 3 is used to compare amplitude-modulated signals created according to 
    their carrier frequency and modulation frequency to the article results.
    The test are done with 50% overlapping time windows as described in the article.
    One .png compliance plot is generated.

    Parameters
    ----------
    None

    Outputs
    -------
    None
    """
   
    # Fixed parameters definition for signals generation
    duration = 3
    mod = 1
    p0 = 1
    fmod = np.arange(0,165,5)
    
    # Overlapping definition for roughness calculation
    overlap = 0.5

    # Initialization with a carrier frequency of 125 Hz
    test_signal = {   "carrier_frequency": 125,
                      "R_file": r"mosqito\tests\roughness\data\Test_fc_125.xlsx" }
            
    # Roughness value for each modulation frequency
    R = np.zeros([fmod.size])

    audio = Audio_signal()
    audio.is_stationary = True
    audio.fs = 48000
    for ind_fmod in range(fmod.size):     
        audio.signal = signal_generation(duration, 125, fmod[ind_fmod], mod, p0)  
        audio.comp_roughness(overlap)
        R[ind_fmod] = np.mean(audio.R)
        
    tst = check_compliance(R, test_signal)
    assert tst
    

# pytest.mark.parametrize allows to execute a test for different data : see http://doc.pytest.org/en/latest/parametrize.html
@pytest.mark.roughness # to skip or run only roughness tests
@pytest.mark.parametrize("signal",
    [{   "carrier_frequency": 125,
         "R_file": r"mosqito\tests\roughness\data\Test_fc_125.xlsx" },
     {   "carrier_frequency": 250,  
         "R_file": r"mosqito\tests\roughness\data\Test_fc_250.xlsx" },
     {   "carrier_frequency": 500,   
         "R_file": r"mosqito\tests\roughness\data\Test_fc_500.xlsx" },
     {   "carrier_frequency": 1000, 
         "R_file": r"mosqito\tests\roughness\data\Test_fc_1000.xlsx" },
     {   "carrier_frequency": 2000,  
         "R_file": r"mosqito\tests\roughness\data\Test_fc_2000.xlsx" },
     {   "carrier_frequency": 4000,  
         "R_file": r"mosqito\tests\roughness\data\Test_fc_4000.xlsx" },
     {   "carrier_frequency": 8000, 
         "R_file": r"mosqito\tests\roughness\data\Test_fc_8000.xlsx" }
     ]
      )


def check_compliance(R, article_ref):
    """Check the compliance of roughness calc. to Daniel and Weber article
    "Psychoacoustical roughness: implementation of an optimized model", 1997.

    Check the compliance of the input data R to figure 3 of the article 
    using the reference data described in the dictionary article_ref.

    Parameters
    ----------
    R : float
        Calculated roughness [asper]
    article_ref : dict
        {   "carrier_frequency": <Path to reference input signal>,
            "R_file": <Path to reference calculated roughness>  }
        
        Dictionary containing link to ref. data
        
        
    Outputs
    -------
    tst : bool
        Compliance to the reference data
    """
    
    # Load reference inputs
    R_article = np.genfromtxt(article_ref["R_file"], skip_header=1)
    
    # Test for comformance (1% tolerance)

    tst = (   R.all() >= R_article.all() * 0.99
          and R.all() <= R_article.all() * 1.01   )
           
    
    # Define and plot the tolerance curves 
    fmod_axis = np.linspace(0,160,33)
    tol_curve_min = np.amin([R_article * 0.99, R_article - 0.1], axis=0)
    tol_curve_min[tol_curve_min < 0] = 0
    tol_curve_max = np.amax([R_article * 1.01, R_article + 0.1], axis=0)
    plt.plot(bark_axis, tol_curve_min, color='red', linestyle = 'solid', label='1% tolerance', linewidth=1)  
    plt.plot(bark_axis, tol_curve_max, color='red', linestyle = 'solid', label='', linewidth=1) 
    plt.legend()
    
    # Compliance plot
    
    plt.plot(fmod_axis, R, label="MoSQITo")    
    if tst_specif:
        plt.text(0.5, 0.5, 'Test passed (1% tolerance not exceeded)', horizontalalignment='center',
        verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='green', alpha=0.3))
    else:
        tst = 0
        plt.text(0.5, 0.5, 'Test not passed', horizontalalignment='center',
        verticalalignment='center', transform=plt.gca().transAxes, 
        bbox=dict(facecolor='red', alpha=0.3))
                
    if tst_N:
        clr = "green"
    else:
        clr = "red"
    plt.title("R = " + str(R) + " asper (Daniel and Weber ref. " + str(R_article) + " asper)", color=clr)
    file_name = "_".join(article_ref["R_file"].split(" "))   
    plt.xlabel("Modulation frequency [Hertz]")
    plt.ylabel("Roughness, [Asper]")
    file_name = "_".join(iso_ref["data_file"].split(" "))
    plt.savefig(
        r"mosqito\tests\roughness\output\test_roughness"
        + file_name.split("/")[-1][:-4]
        + ".png",
        format="png",)
    plt.clf()
    return tst


# test de la fonction
if __name__ == "__main__":
    test_roughness()