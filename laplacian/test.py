import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.signal import savgol_filter

def laplacian_filter(data):
    """
    应用拉普拉斯滤波器

    参数:
    data: 输入信号

    返回:
    filtered_data: 滤波后的数据
    """
    filtered_data = laplace(data)
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
    print(f'Final Mean Squared Error (MSE): {mse}')
    print(f'Final Signal-to-Noise Ratio (SNR): {snr} dB')
    return mse, snr

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
    plt.plot(time, filtered_signal, label='Filtered Signal', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()

def optimize_laplacian_filter(input_file, output_file, column_name='PV', window_length_range=(5, 31), polyorder=2, num_windows=5):
    # 读取数据
    df = pd.read_csv(input_file)
    original_data = df[column_name].values

    # 生成多个窗口长度值
    window_length1 = 5
    window_length2 = 11
    window_length3 = 17

     # 平滑处理
    smoothed_data1 = savgol_filter(original_data, window_length1, polyorder)
    smoothed_data2 = savgol_filter(original_data, window_length2, polyorder)
    smoothed_data3 = savgol_filter(original_data, window_length3, polyorder)

    # 应用拉普拉斯滤波器
    filtered_data1 = laplacian_filter(smoothed_data1)
    filtered_data2 = laplacian_filter(smoothed_data2)
    filtered_data3 = laplacian_filter(smoothed_data3)

    # 绘制原始信号和滤波后信号
    time = np.arange(len(original_data))
    plt.figure(figsize=(10, 6))
    plt.plot(time, original_data, label='Original Signal', linestyle='--', marker='o')
    plt.plot(time, filtered_data1, label='Filtered data3', alpha = 0.5)
    plt.plot(time, filtered_data2, label='Filtered data2', alpha = 0.5)
    plt.plot(time, filtered_data3, label='Filtered data3', alpha = 0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Original data and Filtered data')
    plt.grid()
    plt.savefig('./laplacian/Comparision.jpg')
    plt.show()

    # 评估滤波结果
    evaluate_filter_performance(original_data, filtered_data1)
    evaluate_filter_performance(original_data, filtered_data2)
    evaluate_filter_performance(original_data, filtered_data3)


# 使用示例
input_file = 'loop01.csv'  # 输入文件名
output_file = 'optimized_laplacian_filtered_data.csv'  # 输出文件名
optimize_laplacian_filter(input_file, output_file, column_name='PV', window_length_range=(5, 31), polyorder=2, num_windows=5)
