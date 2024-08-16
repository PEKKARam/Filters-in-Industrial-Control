import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import mean_squared_error

# statsmodels
def evaluate_filtering(original_data, filtered_data):
    """
    评估均值迁移滤波的性能和效果

    参数:
    original_data：初始数据
    filtered_data：滤波处理后的数据

    评估指标:
    mse：均方误差评估滤波效果，该值越小，表示滤波效果越好
    snr：信噪比，SNR 越高，表示滤波后的信号质量越好。
    绘制原始数据和滤波后数据的图形进行对比，直观地评估滤波效果
    """
    # 均方误差
    mse = mean_squared_error(original_data, filtered_data)
    print(f"Mean Squared Error: {mse:.2f}")

    # 信噪比
    def signal_to_noise_ratio(original_data, filtered_data):
        signal_power = np.mean(original_data ** 2)
        noise_power = np.mean((original_data - filtered_data) ** 2)
        return 10 * np.log10(signal_power / noise_power)

    snr = signal_to_noise_ratio(original_data, filtered_data)
    print(f"Signal-to-Noise Ratio: {snr:.2f} dB")

    # 可视化对比
    plt.figure(figsize=(10, 5))
    plt.plot(original_data, label='Original Data', alpha=0.5)
    plt.plot(filtered_data, label='Filtered Data', alpha=0.8)
    plt.legend()
    plt.title('Comparison of Original and Filtered Data')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.savefig('./average/Comparision.jpg')
    plt.show()

def mean_shift_filtering(input_file_path, data_column, output_file_path, bandwidth=None):
    """
    对CSV文件中的指定列数据进行均值迁移滤波，

    参数:
    input_file_path (str): 输入的CSV文件路径
    data_column (str): 需要滤波的列名
    output_file_path (str): 输出的CSV文件路径
    bandwidth (float): 带宽参数，控制搜索窗口的半径

    返回:
    filtered_data：滤波处理后的数据
    data： 转化的二维数组数据
    """
    # 读取CSV文件
    df = pd.read_csv(input_file_path)

    # 提取数据并转换为二维数组
    data = df[[data_column]].values

    # 自动估计带宽
    if bandwidth is None:
        bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
        print(f"Estimated bandwidth: {bandwidth}")

    # 进行均值迁移滤波
    mean_shift = MeanShift(bandwidth=bandwidth)
    mean_shift.fit(data)
    filtered_data = mean_shift.cluster_centers_[mean_shift.labels_]

    # 将滤波后的数据添加到原DataFrame中
    df['filtered_' + data_column] = filtered_data

    # 保存结果到新的CSV文件
    df.to_csv(output_file_path, index=False)

    return filtered_data,data

input_file_path = 'loop01.csv'  # 输入的文件路径
data_column = 'PV'  # 需要滤波的列名
output_file_path = 'average_filtered_data.csv'  # 输出的文件路径

filtered_data,data=mean_shift_filtering(input_file_path, data_column, output_file_path, bandwidth=None)
evaluate_filtering(data, filtered_data)
