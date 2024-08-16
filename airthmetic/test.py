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
    window_size1=4
    window_size2=5
    window_size3=6
    filtered_data1 = np.copy(original_data)
    for i in range(len(original_data)):
        start_index = max(0, i - window_size1 // 2)
        end_index = min(len(original_data), i + window_size // 2 + 1)
        filtered_data1[i] = np.mean(original_data[start_index:end_index])

    filtered_data2 = np.copy(original_data)
    for i in range(len(original_data)):
        start_index = max(0, i - window_size2 // 2)
        end_index = min(len(original_data), i + window_size // 2 + 1)
        filtered_data2[i] = np.mean(original_data[start_index:end_index])

    filtered_data3 = np.copy(original_data)
    for i in range(len(original_data)):
        start_index = max(0, i - window_size3 // 2)
        end_index = min(len(original_data), i + window_size // 2 + 1)
        filtered_data3[i] = np.mean(original_data[start_index:end_index])

    filtered_data4 = np.copy(original_data)
    filtered_data4 = weighted_mean_filter(filtered_data4, window_size2)
    plt.figure(figsize=(12, 6))
    plt.plot(original_data, label='Original Data', alpha=0.8)
    # plt.plot(filtered_data1, label='Filtered1 Data size=4', alpha=0.5)
    plt.plot(filtered_data2, label='Filtered2 Data size=5', alpha=0.5)
    plt.plot(filtered_data3, label='Filtered3 Data size=6', alpha=0.5)
    plt.plot(filtered_data4, label='Filtered4 Data weight', alpha=0.5)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.title(title)
    plt.savefig('./airthmetic/Comparision.jpg')
    plt.show()

    return filtered_data1,filtered_data2,filtered_data3,filtered_data4

# 加权平均滤波函数
def weighted_mean_filter(signal, window_size):
    weights = np.arange(1, window_size + 1)
    weights = np.concatenate([weights, weights[::-1][1:]])
    filtered_signal = np.copy(signal)
    for i in range(len(signal)):
        start_index = max(0, i - window_size // 2)
        end_index = min(len(signal), i + window_size // 2 + 1)
        window = signal[start_index:end_index]
        effective_weights = weights[window_size // 2 - (i - start_index): window_size // 2 + (end_index - i)]
        filtered_signal[i] = np.sum(window * effective_weights) / np.sum(effective_weights)
    return filtered_signal

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
    filtered_data1,filtered_data2,filtered_data3,filtered_data4 = arithmetic_mean_filter(original_data, window_size)
    # df[f'Filtered_{column_name}'] = filtered_data

    # # 保存数据
    # df.to_csv(output_file, index=False)
    # print(f'滤波后的数据已保存到 {output_file}')

    #评估滤波结果
    # evaluate_filter_performance(original_data, filtered_data1)
    evaluate_filter_performance(original_data, filtered_data2)
    evaluate_filter_performance(original_data, filtered_data3)
    evaluate_filter_performance(original_data, filtered_data4)


# 使用示例
input_file = 'loop01.csv'  # 输入文件名
output_file = 'airthmetic_mean_filter_data.csv'  # 输出文件名
main(input_file, output_file, column_name='PV', window_size=5)
