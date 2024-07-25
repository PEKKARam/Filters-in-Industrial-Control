import numpy as np
import matplotlib.pyplot as plt

def filter_evaluate_mse(original_signal, filtered_signal):
    """
    Evaluate the Mean Squared Error (MSE) between the original and filtered signals.
    """
    mse = np.mean((np.array(original_signal) - np.array(filtered_signal))**2)
    return mse

def plot_comparison(original_signal, filtered_signals, alpha_values, title):
    """
    Plot the comparison between the original signal and filtered signals.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(original_signal, label='Original Signal')
    for filtered_signal, alpha in zip(filtered_signals, alpha_values):
        plt.plot(filtered_signal, label=f'Filtered Signal, alpha={alpha}', linestyle='--')
    plt.title(f'Comparison of {title} Filter Effects')
    plt.xlabel('Sample Points')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()