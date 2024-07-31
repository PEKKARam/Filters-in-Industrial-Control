import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filter_package import filter_evaluate_mse, plot_comparison

class FIPFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def set_alpha(self, alpha):
        self.alpha = alpha

    def apply(self, data):
        filtered_data = np.zeros_like(data)
        filtered_data[0] = data[0]  # 初始化第一个值
        for i in range(1, len(data)):
            filtered_data[i] = self.alpha * data[i] + (1 - self.alpha) * filtered_data[i - 1]
        return filtered_data

# 从CSV文件中读取数据
data_df = pd.read_csv('data.csv')
data = data_df['PV'].values  # 假设CSV文件中有一列名为'signal'

# 初始化FIP滤波器
fip_filter = FIPFilter()

# 定义不同的alpha值
alpha_values = [0.1, 0.5, 0.9]

# 对数据进行滤波并计算均方误差
filtered_signals = []
mse_values = []

for alpha in alpha_values:
    fip_filter.set_alpha(alpha)  # 假设FIPFilter有一个设置alpha值的方法
    filtered_signal = fip_filter.apply(data)
    filtered_signals.append(filtered_signal)
    mse = filter_evaluate_mse(data, filtered_signal)
    mse_values.append(mse)
    print(f'Alpha: {alpha}, MSE: {mse}')

# 使用plot_comparison函数绘图
plot_comparison(data, filtered_signals, alpha_values, 'FIP')