import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

def iir_filter(original_data, order, cutoff, fs, filter_type):
    """
    设计IIR滤波器，实现IIR数字滤波

    参数:
    original_data：输出信号
    order: 滤波器的阶数
    cutoff: 截止频率
    fs: 采样频率
    filter_type: 滤波器类型 ('low', 'high', 'bandpass', 'bandstop')

    返回:
    filtered_data: 滤波后的数据
    """
    # nyquist = 0.5 * fs
    # normal_cutoff = cutoff / nyquist
    # b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    # filtered_data = lfilter(b, a, original_data)

    nyquist = 0.5 * fs
    if isinstance(cutoff, list):  # 如果cutoff是列表，说明是带通或带阻滤波器
        normal_cutoff = [freq / nyquist for freq in cutoff]
    else:
        normal_cutoff = cutoff / nyquist

    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_data = lfilter(b, a, original_data)
    # # 绘制频率响应
    # w, h = freqz(b, a, worN=8000)
    # plt.figure()
    # plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
    # plt.plot(cutoff, 0.5 * np.sqrt(2), 'ko')
    # plt.axvline(cutoff, color='k')
    # plt.xlim(0, 0.5 * fs)
    # plt.title("Filter Frequency Response")
    # plt.xlabel('Frequency [Hz]')
    # plt.grid()
    # plt.show()

    return filtered_data

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

def main(input_file, output_file, column_name='PV', order=4, cutoff=100, fs=1000, filter_type='low'):
    # 读取数据
    df = pd.read_csv(input_file)
    original_data = df[column_name].values

    order1=3
    order2=4
    order3=5
    cutoff1=[50, 150]
    # IIR滤波器
    filtered_data1=iir_filter(original_data,order, cutoff, fs, filter_type)
    filtered_data2=iir_filter(original_data,order, cutoff, fs, 'high')
    filtered_data3=iir_filter(original_data,order, cutoff1, fs, 'bandpass')
    # df[f'Filtered_{column_name}'] = filtered_data

    # # 保存滤波后的数据
    # df.to_csv(output_file, index=False)
    # print(f'滤波后的数据已保存到 {output_file}')

    # 评估滤波结果
    evaluate_filter_performance(original_data, filtered_data1)
    evaluate_filter_performance(original_data, filtered_data2)
    evaluate_filter_performance(original_data, filtered_data3)

    # 绘制信号
    # plot_signal(np.arange(len(original_data)), original_data, filtered_data)
    time=np.arange(len(original_data))
    signal=original_data
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal, label='Original Signal', linestyle='--', marker='o')
    plt.plot(time, filtered_data1, label='Filtered data1 low',alpha=0.5)
    plt.plot(time, filtered_data2, label='Filtered data2 high', alpha=0.5)
    plt.plot(time, filtered_data3, label='Filtered data3 bandpass', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Signal and Filtered Signal')
    plt.grid()
    plt.savefig('./IIR/Comparision.jpg')
    plt.show()


# 使用示例
input_file = 'loop01.csv'  # 输入文件名
output_file = 'iir_filtered_data.csv'  # 输出文件名
main(input_file, output_file, column_name='PV', order=4, cutoff=100, fs=1000, filter_type='low')
