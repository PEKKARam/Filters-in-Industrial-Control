import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import laplace

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
    plt.savefig('./laplacian/Comparision1.jpg')
    plt.show()

def main(input_file, output_file, column_name='PV'):
    # 读取数据
    df = pd.read_csv(input_file)
    original_data = df[column_name].values

    # 应用拉普拉斯滤波器
    filtered_data = laplacian_filter(original_data)
    df[f'Filtered_{column_name}'] = filtered_data

    # 保存滤波后的数据
    df.to_csv(output_file, index=False)
    print(f'滤波后的数据已保存到 {output_file}')

    # 评估滤波结果
    mse, snr = evaluate_filter_performance(original_data, filtered_data)
    print(f'Mean Squared Error (MSE): {mse}')
    print(f'Signal-to-Noise Ratio (SNR): {snr} dB')

    # 绘制信号
    plot_signal(np.arange(len(original_data)), original_data, filtered_data)

# 使用示例
input_file = 'loop01.csv'  # 输入文件名
output_file = 'laplacian_filtered_data.csv'  # 输出文件名
main(input_file, output_file, column_name='PV')
