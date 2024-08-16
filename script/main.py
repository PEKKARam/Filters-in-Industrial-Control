import os
import pandas as pd
import matplotlib.pyplot as plt
from filters import get_filter_function

# 设置参数
filter_name = 'Kalman'  # 切换不同的滤波器名称

# 获取滤波函数
filter_function = get_filter_function(filter_name)

# 文件夹路径
folder_path = '滤波测试数据'

# 遍历文件夹中的所有CSV文件
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        
        # 读取CSV文件
        data = pd.read_csv(file_path)
        
        # 提取PV列
        measurements = data['PV'].values
        
        # 应用选定的滤波器
        filtered_signal = filter_function(measurements)
        
        # 添加滤波结果到新列中
        column_name = f'Filtered_PV_{filter_name}'
        data[column_name] = filtered_signal
        
        # 将更新后的数据保存回原CSV文件
        data.to_csv(file_path, index=False)
