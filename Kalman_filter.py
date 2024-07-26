import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, process_variance, measurement_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

        return self.posteri_estimate

# 读取CSV文件
data = pd.read_csv('loop01.csv')

# 提取PV列
measurements = data['PV'].values

# 卡尔曼滤波器参数
process_variance = 0.01  # 过程噪声协方差
measurement_variance = 1  # 测量噪声协方差
estimated_measurement_variance = 1  # 初始估计的方差

# 初始化滤波器
kf = KalmanFilter(process_variance, measurement_variance, estimated_measurement_variance)

# 存储滤波结果
filtered_signal = np.zeros_like(measurements)

for i in range(len(measurements)):
    filtered_signal[i] = kf.update(measurements[i])

# 将滤波结果保存到新的CSV文件
filtered_data = data.copy()
filtered_data['Filtered_PV'] = filtered_signal
filtered_data.to_csv('filtered_loop001.csv', index=False)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(data['PV'], label='0riginal data', color='red')
plt.plot(filtered_signal, label='filtered data', color='blue')
plt.legend()
plt.xlabel('Index')
plt.ylabel('PV')
plt.title('Kalman Filter')
plt.show()
