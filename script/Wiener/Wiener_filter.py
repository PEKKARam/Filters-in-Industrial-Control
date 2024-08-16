from scipy.signal import wiener

def apply_wiener_filter(signal, mysize=None, noise=None):
    """
    应用维纳滤波器到信号
    :param signal: 输入信号 (数组)
    :param mysize: 用于滤波器的窗口尺寸 (可选)
    :param noise: 噪声功率估计值 (可选)
    :return: 滤波后的信号
    """
    return wiener(signal, mysize=mysize, noise=noise)
