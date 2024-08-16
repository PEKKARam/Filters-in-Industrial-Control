import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def arithmetic_mean_filter(original_data, window_size,title='Original and Filtered Data'):
    """
    对数据进行均值迁移滤波，

    参数:
    original_data: 初始数据
    window_size: 滤波窗口大小

    输出:
    绘制原始数据和滤波后数据的对比图

    返回:
    filtered_data：滤波处理后的数据
    """
    filtered_data = np.copy(original_data)
    for i in range(len(original_data)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(original_data), i + window_size // 2 + 1)
        filtered_data[i] = np.mean(original_data[start_index:end_index])

    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data', linestyle='--', marker='o')
    plt.plot(filtered_data, label='Filtered Data', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title(title)
    plt.show()
    return filtered_data

def evaluate_filter_performance(original_data, filtered_data):
    """
    评估均值迁移滤波

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


def main(input_file, output_file, column_name='PV', window_size=5):
    # 读取数据
    df = pd.read_csv(input_file)
    original_data = df[column_name].values

    # 应用滤波
    filtered_data = arithmetic_mean_filter(original_data, window_size)
    df[f'Filtered_{column_name}'] = filtered_data

    # 保存数据
    df.to_csv(output_file, index=False)
    print(f'滤波后的数据已保存到 {output_file}')

    #评估滤波结果
    evaluate_filter_performance(original_data, filtered_data)


# 使用示例
input_file = 'loop01.csv'  # 输入文件名
output_file = 'airthmetic_mean_filter_data.csv'  # 输出文件名
main(input_file, output_file, column_name='PV', window_size=5)
