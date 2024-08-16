import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'loop01.csv'
data = pd.read_csv(file_path)

# 提取原始数据和滤波后的数据
original_data = data['PV']
filtered_data_kalman = data['Filtered_PV_Kalman']
filtered_data_median = data['Filtered_PV_Median']
filtered_data_wiener = data['Filtered_PV_Wiener']
filtered_data_chebyshev = data['Filtered_PV_Chebyshev']

# 创建图像
plt.figure(figsize=(14, 8))

# 绘制原始数据
plt.plot(original_data, label='Original PV', linestyle='dotted', color='r')

# 绘制滤波后的数据
if 'Filtered_PV_Kalman' in data.columns:
    plt.plot(filtered_data_kalman, label='Kalman Filtered PV', color='b')
if 'Filtered_PV_Median' in data.columns:
    plt.plot(filtered_data_median, label='Median Filtered PV', color='g')
if 'Filtered_PV_Wiener' in data.columns:
    plt.plot(filtered_data_wiener, label='Wiener Filtered PV', color='c')
if 'Filtered_PV_Chebyshev' in data.columns:
    plt.plot(filtered_data_chebyshev, label='Chebyshev Filtered PV', color='m')

# 设置图像属性
plt.legend()
plt.xlabel('Index')
plt.ylabel('PV')
plt.title('Comparison of Filtering Effects on loop01.csv')
plt.grid(True)
plt.show()
