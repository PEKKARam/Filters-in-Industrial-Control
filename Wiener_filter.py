import numpy as np
from scipy import signal
from filter_evaluater import generate_noisy_data, evaluate_filter, plot_results

def filter_wiener(noisy_data, original_data, kernel_size=5):
    """
    应用维纳滤波器到带噪声数据上。
    
    参数:
    noisy_data (np.ndarray): 带噪声的数据。
    original_data (np.ndarray): 原始无噪声的数据。
    kernel_size (int): 滤波器的窗口大小。
    
    返回:
    np.ndarray: 滤波后的数据。
    """
    # 计算原始数据和带噪声数据的功率谱密度
    noisy_psd = np.abs(np.fft.fft(noisy_data)) ** 2
    original_psd = np.abs(np.fft.fft(original_data)) ** 2
    
    # 计算维纳滤波器的频率响应
    wiener_filter_response = original_psd / (original_psd + noisy_psd)
    
    # 应用滤波器到频域
    noisy_freq = np.fft.fft(noisy_data)
    filtered_freq = noisy_freq * wiener_filter_response
    filtered_data = np.fft.ifft(filtered_freq).real
    
    return filtered_data

# 示例用法
if __name__ == "__main__":
    # 生成示例数据
    time, original_data, noisy_data = generate_noisy_data(seed=0, num_samples=500, noise_std=0.5)
    
    # 应用维纳滤波器
    kernel_size = 5  # 虽然维纳滤波器理论上不依赖窗口大小，但可以传递参数以便其他滤波器使用
    filtered_data = filter_wiener(noisy_data, original_data, kernel_size)
    
    # 评估滤波器性能
    psnr = evaluate_filter(original_data, filtered_data)
    
    # 绘制结果
    plot_results(time, original_data, noisy_data, filtered_data, psnr)
