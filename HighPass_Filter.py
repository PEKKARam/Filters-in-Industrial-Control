from filter_package import *
import numpy as np
import matplotlib.pyplot as plt

def filter_high_pass(signal, alpha):
    """
    Apply a first-order high-pass filter to a signal.
    :param signal: Input signal (list or numpy array).
    :param alpha: Filter coefficient (0 < alpha < 1).
    :return: Filtered signal.
    """
    filtered_signal = np.zeros_like(signal)
    for i in range(1, len(signal)):
        filtered_signal[i] = alpha * (filtered_signal[i-1] + signal[i] - signal[i-1])
    return filtered_signal

# Example usage
time = np.linspace(0, 10, 100)
signal = np.sin(time) + np.random.normal(0, 0.5, len(time))

# Apply the filter
alpha_values = [0.1, 0.3, 0.5]
filtered_signals = [filter_high_pass(signal, alpha) for alpha in alpha_values]

# Plot comparison
plot_comparison(signal, filtered_signals, alpha_values)