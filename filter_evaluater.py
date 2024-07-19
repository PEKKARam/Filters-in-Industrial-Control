import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

def generate_noisy_data(seed=0, num_samples=500, noise_std=0.5):
    """
    生成带噪声的数据。
    
    参数:
    seed (int): 随机种子。
    num_samples (int): 数据样本数。
    noise_std (float): 噪声标准差。
    
    返回:
    tuple: (时间序列, 原始数据, 带噪声数据)
    """
    np.random.seed(seed)  # 设置随机种子以保证结果可重复
    time = np.linspace(0, 1, num_samples)
    original_data = np.sin(2 * np.pi * 5 * time)  # 原始信号（正弦波）
    noise = np.random.normal(0, noise_std, original_data.shape)  # 噪声
    noisy_data = original_data + noise  # 带噪声的信号
    
    return time, original_data, noisy_data

def evaluate_filter(original_data, filtered_data):
    """
    评估滤波器的性能。
    
    参数:
    original_data (np.ndarray): 原始输入数据数组。
    filtered_data (np.ndarray): 滤波后的数据数组。
    
    返回:
    float: 评估指标（PSNR）。
    """
    # 计算均方误差 (MSE)
    mse = np.mean((original_data - filtered_data) ** 2)
    
    if mse == 0:
        return float('inf')
    
    # 计算峰值信噪比 (PSNR)
    max_pixel = np.max(original_data)
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def plot_results(time, original_data, noisy_data, filtered_data, psnr):
    """
    绘制原始数据、带噪声数据和滤波后数据的对比图。
    
    参数:
    time (np.ndarray): 时间序列数据。
    original_data (np.ndarray): 原始数据。
    noisy_data (np.ndarray): 带噪声数据。
    filtered_data (np.ndarray): 滤波后的数据。
    psnr (float): 峰值信噪比。
    """
    plt.figure(figsize=(15, 5))
    
    plt.plot(time, original_data, label='original data', color='blue')
    plt.plot(time, noisy_data, label='noisy data', color='red', linestyle='dashed')
    plt.plot(time, filtered_data, label='filtered data', color='green')
    
    
    plt.legend()
    plt.title(f'filtering effect (PSNR_1: {psnr:.2f} dB)')
    plt.xlabel('time')
    plt.ylabel('signal range')
    plt.show()
