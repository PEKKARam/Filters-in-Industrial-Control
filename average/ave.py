import pandas as pd
import numpy as np
import time
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tracemalloc

# pandas numpy time scikit-learn matplotlib tracemalloc
def evaluate_mean_shift(input_file_path, data_column, output_file_path, bandwidth=None):
    """
    评估均值迁移滤波的性能和效果

    参数:
    input_file_path (str): 输入的CSV文件路径
    data_column (str): 需要滤波的列名
    output_file_path (str): 输出的CSV文件路径
    bandwidth (float): 带宽参数，控制搜索窗口的半径

    评估指标:
    time：记录算法运行时间
    current_memory，peak_memory：记录当前内存和峰值内存
    mse：均方误差评估滤波效果，该值越小，表示滤波效果越好
    """
    # 开始性能测量
    start_time = time.time()
    tracemalloc.start()

    # 进行均值迁移滤波
    df,filtered_data,data = mean_shift_filtering(input_file_path, data_column, output_file_path, bandwidth=None)

    # 结束性能测量
    end_time = time.time()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 计算评估指标
    mse = mean_squared_error(data, filtered_data)
    time_taken = end_time - start_time
    current_memory = current / 10**6
    peak_memory = peak / 10**6

    # 打印性能指标
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Current memory usage: {current_memory:.2f} MB")
    print(f"Peak memory usage: {peak_memory:.2f} MB")
    print(f"Mean Squared Error: {mse:.2f}")

    # 可视化原始数据和滤波后的数据
    plt.figure(figsize=(10, 5))
    plt.plot(df[data_column], label='Original Data', alpha=0.5)
    plt.plot(df['filtered_' + data_column], label='Filtered Data', alpha=0.8)
    plt.legend()
    plt.title('Mean Shift Filtering')
    plt.xlabel('Index')
    plt.ylabel('Value')
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
    df: 带有滤波后数据的DataFrame
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

    return df,filtered_data,data

input_file_path = 'loop01.csv'  # 输入的文件路径
data_column = 'PV'  # 需要滤波的列名
output_file_path = 'average_filtered_data.csv'  # 输出的文件路径

evaluate_mean_shift(input_file_path, data_column, output_file_path, bandwidth=None)
