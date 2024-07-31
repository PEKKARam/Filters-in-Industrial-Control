import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import ellip, lfilter
from filter_package import filter_evaluate_mse, plot_comparison

class EllipticFilter:
    def __init__(self, order=5, rp=1, rs=40, Wn=0.1):
        self.order = order
        self.rp = rp
        self.rs = rs
        self.Wn = Wn
        self.b, self.a = ellip(self.order, self.rp, self.rs, self.Wn)

    def set_params(self, order=None, rp=None, rs=None, Wn=None):
        if order is not None:
            self.order = order
        if rp is not None:
            self.rp = rp
        if rs is not None:
            self.rs = rs
        if Wn is not None:
            self.Wn = Wn
        self.b, self.a = ellip(self.order, self.rp, self.rs, self.Wn)

    def apply(self, data):
        return lfilter(self.b, self.a, data)

# 从CSV文件中读取数据
data_df = pd.read_csv('data.csv')
data = data_df['PV'].values  # 假设CSV文件中有一列名为'signal'

# 初始化椭圆滤波器
elliptic_filter = EllipticFilter()

# 定义不同的滤波器参数
filter_params = [
    {'order': 3, 'rp': 1, 'rs': 30, 'Wn': 0.1},
    {'order': 5, 'rp': 1, 'rs': 40, 'Wn': 0.1},
    {'order': 7, 'rp': 1, 'rs': 50, 'Wn': 0.1}
]

# 对数据进行滤波并计算均方误差
filtered_signals = []
mse_values = []

for params in filter_params:
    elliptic_filter.set_params(**params)
    filtered_signal = elliptic_filter.apply(data)
    filtered_signals.append(filtered_signal)
    mse = filter_evaluate_mse(data, filtered_signal)
    mse_values.append(mse)
    print(f'Params: {params}, MSE: {mse}')

# 使用plot_comparison函数绘图
plot_comparison(data, filtered_signals, [str(params) for params in filter_params], 'Elliptic')
