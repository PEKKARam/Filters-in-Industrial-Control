import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

def gaussian_filter(data, sigma):
    """
    应用高斯滤波器

    参数:
    data: 输入信号
    sigma: 高斯滤波器的标准差

    返回:
    filtered_data: 滤波后的数据
    """
    filtered_data = gaussian_filter1d(data, sigma)
    return filtered_data

def evaluate_filter_performance(original_data, filtered_data):
    """
    评估滤波性能

    参数:
    original_data: 初始数据
    filtered_data: 滤波后数据

    输出:
    MSE：均方误差
    SNR：信噪比
    """
    mse = np.mean((original_data - filtered_data) ** 2)
    signal_power = np.mean(original_data ** 2)
    noise_power = np.mean((original_data - filtered_data) ** 2)
    snr = 10 * np.log10(signal_power / noise_power)
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Signal-to-Noise Ratio (SNR): {snr} dB')

def plot_signal(time, signal, filtered_signal, title='Signal and Filtered Signal'):
    """
    绘制原始信号和滤波后信号

    参数:
    time: 时间序列
    signal: 原始信号
    filtered_signal: 滤波后信号
    title: 图标题
    """
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label='Original Signal', linestyle='--', marker='o')
    plt.plot(time, filtered_signal, label='Filtered Signal', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()

def optimize_gaussian_filter(input_file, output_file, column_name='PV', sigma_range=(0.5, 5.0), num_sigma=10):
    # 读取数据
    df = pd.read_csv(input_file)
    original_data = df[column_name].values

    filtered_data1 = gaussian_filter(original_data, 0.5)
    filtered_data2 = gaussian_filter(original_data, 1.0)
    filtered_data3 = gaussian_filter(original_data, 1.5)
    evaluate_filter_performance(original_data, filtered_data1)
    evaluate_filter_performance(original_data, filtered_data2)
    evaluate_filter_performance(original_data, filtered_data3)

    # 绘制信号
    # plot_signal(np.arange(len(original_data)), original_data, filtered_data)
    time=np.arange(len(original_data))

    plt.figure(figsize=(10, 6))
    plt.plot(time, original_data, label='Original Signal', linestyle='--', marker='o')
    plt.plot(time, filtered_data1, label='Filtered data1 sigma=0.5', alpha=0.5)
    plt.plot(time, filtered_data2, label='Filtered data2 sigma=1.0', alpha=0.5)
    plt.plot(time, filtered_data3, label='Filtered data3 sigma=1.5', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Signal and Filtered Signal')
    plt.grid()
    plt.savefig('./gaussian/Comparision2.jpg')
    plt.show()

# 使用示例
input_file = 'loop01.csv'  # 输入文件名
output_file = 'gaussian_filtered_data.csv'  # 输出文件名
optimize_gaussian_filter(input_file, output_file, column_name='PV', sigma_range=(0.5, 5.0), num_sigma=10)
