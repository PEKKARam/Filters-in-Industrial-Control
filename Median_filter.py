# median_filter.py
import numpy as np
from filter_evaluater import generate_noisy_data, evaluate_filter, plot_results
from scipy.signal import medfilt

def filter_median(data, kernel_size=3):
    """
    应用中值滤波器到输入数据上。
    
    参数:
    data (np.ndarray): 输入数据数组（信号）。
    kernel_size (int): 滤波器的窗口大小（必须是奇数）。
    
    返回:
    np.ndarray: 滤波后的数据。
    """
    # 使用scipy库中的medfilt函数进行中值滤波
    filtered_data = medfilt(data, kernel_size)
    
    return filtered_data

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    time, original_data, noisy_data = generate_noisy_data(seed=0, num_samples=500, noise_std=0.5)
    
    # 应用中值滤波器
    kernel_size_1 = 5
    kernel_size_2 = 9  # 中值滤波器的窗口大小
    filtered_data_1 = filter_median(noisy_data, kernel_size_1)
    filtered_data_2 = filter_median(noisy_data, kernel_size_2)
    
    # 评估滤波器性能
    psnr_1 = evaluate_filter(original_data, filtered_data_1)
    psnr_2 = evaluate_filter(original_data, filtered_data_2)
    
    # 绘制结果
    plot_results(time, original_data, noisy_data, filtered_data_1, psnr_1)
    plot_results(time, original_data, noisy_data, filtered_data_2, psnr_2)
