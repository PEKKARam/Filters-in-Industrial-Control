import numpy as np
from scipy.signal import cheby1, filtfilt

def apply_chebyshev_filter(data, fs=1.0, cutoff=0.1, rp=0.1, order=4):
    """
    Apply a Chebyshev filter to the input data.
    
    :param data: The input signal as a numpy array.
    :param fs: Sampling frequency (Hz).
    :param cutoff: Cutoff frequency (Hz).
    :param rp: Maximum ripple allowed in the passband (dB).
    :param order: Order of the filter.
    :return: Filtered signal as a numpy array.
    """
    # Design the Chebyshev filter
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = cheby1(order, rp, normal_cutoff, btype='low', analog=False)
    
    # Apply the filter
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data
