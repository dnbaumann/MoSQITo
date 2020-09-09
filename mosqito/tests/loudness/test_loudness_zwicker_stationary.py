# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 2020

@author: pc
"""
import sys
sys.path.append('../../..')
import csv

#Standard imports
import numpy as np
import matplotlib.pyplot as plt
import pytest

# Local application imports
from mosqito.Classes.Audio_signal import Audio_signal

@pytest.mark.loudness_zwst  # to skip or run only loudness zwicker stationary tests
def test_loudness_zwicker_3oct():
    """Test function for the loudness calculation of a stationary signal

    Test function for the Audio_signal class "comp_loudness" method with
    third octave band spectrum as input. The input spectrum is 
    provided by ISO 532-1 annex B2, the compliance is assessed 
    according to section 5.1 of the standard. One .png compliance
    plot is generated.

    Parameters
    ----------
    None

    Outputs
    -------
    None
    """
    # Third octave levels and frequencies as inputs for stationary loudness
    # (from ISO 532-1 annex B2)
    test_signal_1 = np.array(
        [
            -60,
            -60,
            78,
            79,
            89,
            72,
            80,
            89,
            75,
            87,
            85,
            79,
            86,
            80,
            71,
            70,
            72,
            71,
            72,
            74,
            69,
            65,
            67,
            77,
            68,
            58,
            45,
            30.0,
        ]
    )
    
    fr = [ 25, 31.5, 40, 50,63, 80,  100, 125,  160,200, 250, 315, 400, 500,
            630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300,
            8000, 10000, 12500,  ] 

    signal = {
        "data_file": "Test signal 1.txt",
        "N": 83.296,
        "N_specif_file": "mosqito/tests/data/ISO_532-1/test_signal_1.csv",
    }
         
    audio = Audio_signal()         
    audio.set_third_oct(test_signal_1, fr)  
    audio.comp_loudness()
    N = audio.N
    N_specific = audio.N_specific   
    tst = check_compliance(N, N_specific, signal)
    assert tst
    
    
    


# pytest.mark.parametrize allows to execute a test for different data : see http://doc.pytest.org/en/latest/parametrize.html
@pytest.mark.loudness_zwst  # to skip or run only loudness zwicker stationary tests
@pytest.mark.parametrize(
    "signal",
    [
        {
            "data_file": "mosqito/tests/data/ISO_532-1/Test signal 2 (250 Hz 80 dB).wav",
            "N": 14.655,
            "N_specif_file": "mosqito/tests/data/ISO_532-1/test_signal_2.csv",
        },
        {
            "data_file": "mosqito/tests/data/ISO_532-1/Test signal 3 (1 kHz 60 dB).wav",
            "N": 4.019,
            "N_specif_file": "mosqito/tests/data/ISO_532-1/test_signal_3.csv",
        },
        {
            "data_file": "mosqito/tests/data/ISO_532-1/Test signal 4 (4 kHz 40 dB).wav",
            "N": 1.549,
            "N_specif_file": "mosqito/tests/data/ISO_532-1/test_signal_4.csv",
        },
        {
            "data_file": "mosqito/tests/data/ISO_532-1/Test signal 5 (pinknoise 60 dB).wav",
            "N": 10.498,
            "N_specif_file": "mosqito/tests/data/ISO_532-1/test_signal_5.csv",
        },
    ],
)

@pytest.mark.loudness_zwst  # to skip or run only loudness zwicker stationary tests
def test_loudness_zwicker_wav(signal):
    """Test function for the script loudness_zwicker_stationary

    Test function for the script loudness_zwicker_stationary with
    .wav file as input. The input file is provided by ISO 532-1 annex 
    B3, the compliance is assessed according to section 5.1 of the 
    standard. One .png compliance plot is generated.

    Parameters
    ----------
    None

    Outputs
    -------
    None
    """
    # Test signal as input for stationary loudness
    # (from ISO 532-1 annex B3)
    audio = Audio_signal()
    # Load signal and compute third octave band spectrum
    audio.load_wav(True, signal["data_file"], calib=2 * 2 ** 0.5)
    audio.comp_third_oct()
    # Compute Loudness
    audio.comp_loudness()
    N = audio.N
    N_specific = audio.N_specific
    # Check ISO 532-1 compliance
    assert check_compliance(N, N_specific, signal)


def check_compliance(N, N_specific, iso_ref):
    """Check the compliance of loudness calc. to ISO 532-1

    Check the compliance of the input data N and N_specific
    to section 5.1 of ISO 532-1 by using the reference data
    described in dictionary iso_ref.

    Parameters
    ----------
    N : float
        Calculated loudness [sones]
    N_specific : numpy.ndarray
        Specific loudness [sones/bark]
    iso_ref : dict
        {
            "data_file": <Path to reference input signal>,
            "N": <Reference loudness value>,
            "N_specif_file": <Path to reference calculated specific loudness>    
        }
        Dictionary containing link to ref. data

    Outputs
    -------
    tst : bool
        Compliance to the reference data
    """
    # Load ISO reference outputs
    N_iso = iso_ref["N"]
    N_specif_iso = np.genfromtxt(iso_ref["N_specif_file"], skip_header=1)
  
    
    # Test for ISO 532-1 comformance (section 5.1)
    tst_N = (
        N >= N_iso * 0.95
        and N <= N_iso * 1.05
        and N >= N_iso - 0.1
        and N <= N_iso + 0.1
    )
    tst_specif = (
        (N_specific >= np.amin([N_specif_iso * 0.95, N_specif_iso - 0.1], axis=0)).all()
        and (
            N_specific <= np.amax([N_specif_iso * 1.05, N_specif_iso + 0.1], axis=0)
        ).all()
    )
    tst = tst_N and tst_specif
    
    # Generate compliance plot
    bark_axis = np.linspace(0.1, 24, int(24 / 0.1))
    plt.plot(bark_axis, N_specific, label="MoSQITo")
    
    #Formating
    tolerances = [ [0.9, 0.95, 1.05, 1.1], 
                   [-0.2, -0.1, 0.1, 0.2] ]
    style = ['solid', 'dashed', 'dashed', 'solid']
    lab = ['10% tolerance', '5% tolerance', '', '']
    clrs = ['red', 'orange', 'orange', 'red']
    
    # Define the tolerance curves and build compliance matrix
    comp = np.zeros((4,N_specific.size))
    for i in np.arange(4):
            if i in [0, 1]:
                tol_curve = np.amin([N_specif_iso * tolerances[0][i], N_specif_iso + tolerances[1][i]], axis=0)
                comp[i,:] = N >= tol_curve
            else:
                tol_curve = np.amax([N_specif_iso * tolerances[0][i], N_specif_iso + tolerances[1][i]], axis=0)
                comp[i,:] = N <= tol_curve
            tol_curve[tol_curve < 0] = 0
            
            # Plot tolerance curves
            plt.plot(bark_axis, tol_curve, color=clrs[i], linestyle = style[i], label=lab[i], linewidth=1)    
            plt.legend()
# Check compliance
    comp_10 = np.array([comp[0,i] and comp[3,i] for i in np.arange(N_specific.size)])
    comp_5 = np.array([comp[1,i] and comp[2,i] for i in np.arange(N_specific.size)])
    ind_10 = np.nonzero(comp_10 == 0)[0]
    ind_5 = np.nonzero(comp_5 == 0)[0]
    if ind_5.size == 0:
        plt.text(0.5, 0.5, 'Test passed (5% tolerance not exceeded)', horizontalalignment='center',
        verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='green', alpha=0.3))
    elif ind_5.size / N.size <= 0.01: 
        plt.text(0.5, 0.5, 'Test passed (5% tolerance exceeded in maximum 1% of time)', horizontalalignment='center',
        verticalalignment='center', transform=plt.gca().transAxes,
        bbox=dict(facecolor='orange', alpha=0.3), wrap=True)
    else:
        tst = 0
        plt.text(0.5, 0.5, 'Test not passed', horizontalalignment='center',
        verticalalignment='center', transform=plt.gca().transAxes, 
        bbox=dict(facecolor='red', alpha=0.3))
  #
# Highlights non-compliant area
    for i in ind_10:
        plt.axvspan(bark_axis[i]-0.001, bark_axis[i]+0.001, facecolor="red", alpha=0.3)
    for i in ind_5:
        if not i in ind_10:
            plt.axvspan(bark_axis[i]-0.001, bark_axis[i]+0.001, facecolor="orange", alpha=0.3)
               
 
    plt.title("N = " + str(N) + " sone (ISO ref. " + str(N_iso) + " sone)",)    
    plt.xlabel("Critical band rate [Bark]")
    plt.ylabel("Specific loudness, [sone/Bark]")
    file_name = "_".join(iso_ref["data_file"].split(" "))
    plt.savefig(
        r"mosqito\tests\output\test_loudness_zwicker_wav_"
        + file_name.split("/")[-1][:-4]
        + ".png",
        format="png",)
    plt.show()
    plt.clf()
    return tst


# test de la fonction
if __name__ == "__main__":
    test_loudness_zwicker_3oct()
