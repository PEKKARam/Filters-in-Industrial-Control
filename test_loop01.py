import pandas as pd
import matplotlib.pyplot as plt
from filters import get_filter_function

# 设置滤波器名称
filter_name = 'Median'  # 你可以修改这个值来切换不同的滤波器

# 获取滤波函数
filter_function = get_filter_function(filter_name)

# 读取CSV文件
file_path = 'loop01.csv'
data = pd.read_csv(file_path)

# 确保PV列存在
if 'PV' not in data.columns:
    raise ValueError(f"'PV' column not found in {file_path}")

# 提取PV列
measurements = data['PV'].values

# 应用选定的滤波器
filtered_signal = filter_function(measurements)

# 添加滤波结果到新列中
column_name = f'Filtered_PV_{filter_name}'
data[column_name] = filtered_signal

# 保存更新后的数据回原CSV文件
data.to_csv(file_path, index=False)

# 可视化原始和滤波后的数据
plt.figure(figsize=(12, 6))
plt.plot(data['PV'], label='Original PV', linestyle='dotted', color='r')
plt.plot(data[column_name], label=column_name, color='b')
plt.legend()
plt.xlabel('Index')
plt.ylabel('PV')
plt.title(f'{filter_name} Applied to loop01.csv')
plt.show()
