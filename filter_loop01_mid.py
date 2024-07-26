import numpy as np
import pandas as pd
from scipy.signal import medfilt
import matplotlib.pyplot as plt

file_path = 'loop01.csv'
data = pd.read_csv(file_path)

def filter_median(data, kernel_size=3):
    """
    应用中值滤波器到输入数据上。
    
    参数:
    data (np.ndarray): 输入数据数组（信号）。
    kernel_size (int): 滤波器的窗口大小（必须是奇数）。
    
    返回:
    np.ndarray: 滤波后的数据。
    """
    # 使用scipy库中的medfilt函数进行中值滤波
    filtered_data = medfilt(data, kernel_size)
    
    return filtered_data

if __name__ == "__main__":
    kernel_size = 9  # 中值滤波器的窗口大小
    filtered_data = filter_median(data['PV'], kernel_size)
    
    data.to_csv(file_path, index = False)
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(data['PV'], label='original data', color='blue')
    plt.plot(filtered_data, label='filtered data', color='red')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('PV')
    plt.title('Median Filter')
    plt.show()