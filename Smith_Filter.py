import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from filter_package import filter_evaluate_mse, plot_comparison

class SmithFilter:
    def __init__(self, beta=0.5):
        self.beta = beta

    def set_beta(self, beta):
        self.beta = beta

    def apply(self, data):
        filtered_data = np.zeros_like(data)
        filtered_data[0] = data[0]  # 初始化第一个值
        for i in range(1, len(data)):
            filtered_data[i] = self.beta * data[i] + (1 - self.beta) * filtered_data[i - 1]
        return filtered_data

# 从CSV文件中读取数据
data_df = pd.read_csv('data.csv')
data = data_df['PV'].values  # 假设CSV文件中有一列名为'PV'

# 初始化Smith滤波器
smith_filter = SmithFilter()

# 定义不同的beta值
beta_values = [0.1, 0.5, 0.9]

# 对数据进行滤波并计算均方误差
filtered_signals = []
mse_values = []

for beta in beta_values:
    smith_filter.set_beta(beta)
    filtered_signal = smith_filter.apply(data)
    filtered_signals.append(filtered_signal)
    mse = filter_evaluate_mse(data, filtered_signal)
    mse_values.append(mse)
    print(f'Beta: {beta}, MSE: {mse}')

# 使用plot_comparison函数绘图
plot_comparison(data, filtered_signals, beta_values, 'Smith')
