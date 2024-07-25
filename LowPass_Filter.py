from filter_package import *
import numpy as np
import matplotlib.pyplot as plt

def filter_low_pass(input_signal, alpha):
    """
    Implement a first-order low-pass filter.
    """
    output_signal = [input_signal[0]]  # Initialize the output signal list with the first element of the input signal
    for n in range(1, len(input_signal)):
        output_signal.append(alpha * input_signal[n] + (1 - alpha) * output_signal[n-1])
    return output_signal

# Example usage
time = np.linspace(0, 10, 100)
signal = np.sin(time) + np.random.normal(0, 0.5, len(time))

# Apply the filter
alpha_values = [0.1, 0.3, 0.5]
filtered_signals = [filter_low_pass(signal, alpha) for alpha in alpha_values]

# Evaluate filter performance
mse_values = [filter_evaluate_mse(signal, filtered_signal) for filtered_signal in filtered_signals]

for alpha, mse in zip(alpha_values, mse_values):
    print(f"alpha={alpha} MSE: {mse}")

# Plot comparison
plot_comparison(signal, filtered_signals, alpha_values, 'Low Pass')