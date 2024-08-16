import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
from filter_package import *
import pandas as pd

# 设计一个二阶低通滤波器
def butter_lowpass(cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

# 应用滤波器
def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# 从CSV文件中读取数据
data_df = pd.read_csv('data.csv')
data = data_df['PV'].values 

# 示例使用
fs = 500.0  # 采样频率
cutoff = 50.0  # 截止频率

# 生成不同alpha值的滤波信号
alpha_values = [0.1, 0.5, 0.9]
filtered_signals = [butter_lowpass_filter(data, cutoff * alpha, fs) for alpha in alpha_values]

# 使用plot_comparison函数绘图
plot_comparison(data, filtered_signals, alpha_values, 'Second-Order Lowpass')