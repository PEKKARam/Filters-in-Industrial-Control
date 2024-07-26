import numpy as np
import pandas as pd
from scipy.signal import wiener
import matplotlib.pyplot as plt

def load_data(file_path):
    """
    加载数据文件。
    :param file_path: 数据文件路径。
    :return: 加载的数据。
    """
    return pd.read_csv(file_path)

def apply_wiener_filter(data, column_name, mysize=None, noise=None):
    """
    对指定列应用维纳滤波器。
    :param data: 输入数据。
    :param column_name: 要滤波的列名。
    :param mysize: 用于滤波器的窗口大小。
    :param noise: 噪声功率。如果已知，可以提供该值以改进滤波效果。
    :return: 添加滤波结果的新数据。
    """
    # 提取数据列
    column_data = data[column_name].to_numpy()
    
    # 应用维纳滤波器
    filtered_data = wiener(column_data, mysize=mysize, noise=noise)
    
    # 将结果添加到新的列中
    data[f'{column_name}_filtered'] = filtered_data
    return data

def save_data(data, file_path):
    """
    保存处理后的数据到文件。
    :param data: 要保存的数据。
    :param file_path: 输出文件路径。
    """
    data.to_csv(file_path, index=False)

# 使用示例
file_path = 'loop01.csv'
data = load_data(file_path)

# 应用维纳滤波器
filtered_data = apply_wiener_filter(data, column_name='PV')

plt.figure(figsize=(12, 6))
    
plt.plot(data['PV'], label='original data', color='blue')
plt.plot(data['PV_filtered'], label='filtered data', color='red')
plt.legend()
plt.xlabel('Index')
plt.ylabel('PV')
plt.title('Wiener Filter')    
plt.show()
