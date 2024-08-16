import numpy as np
from scipy.ndimage import median_filter

def apply_median_filter(data, size=3):
    return median_filter(data, size=size)
