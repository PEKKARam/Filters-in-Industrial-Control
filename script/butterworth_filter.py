import numpy as np
from scipy.signal import butter, filtfilt

def apply_butterworth_filter(data, cutoff, fs, order=5, filter_type='low'):
    """
    应用巴特沃兹滤波器。

    参数:
    data -- 需要滤波的原始信号
    cutoff -- 截止频率
    fs -- 采样频率
    order -- 滤波器阶数
    filter_type -- 滤波器类型 ('low' 表示低通, 'high' 表示高通, 'band' 表示带通, 'stop' 表示带阻)

    返回:
    filtered_data -- 滤波后的信号
    """
    nyquist = 0.5 * fs  # 奈奎斯特频率
    normal_cutoff = cutoff / nyquist
    
    # 设计巴特沃兹滤波器
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    
    # 应用滤波器
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data
