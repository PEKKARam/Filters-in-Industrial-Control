def get_filter_function(filter_name):
    if filter_name == 'Kalman':
        from Kalman.Kalman_filter import apply_kalman_filter
        return apply_kalman_filter
    elif filter_name == 'Median':
        from Median.Median_filter import apply_median_filter
        return apply_median_filter
    elif filter_name == 'Wiener':
        from Wiener.Wiener_filter import apply_wiener_filter
        return apply_wiener_filter
    elif filter_name == 'Chebyshev':
        from Chebyshev.chebyshev_filter import apply_chebyshev_filter
        return apply_chebyshev_filter
    elif filter_name == 'Butterworth':
        from Butterworth.butterworth_filter import apply_butterworth_filter
        return apply_butterworth_filter
    else:
        raise ValueError(f"Unsupported filter: {filter_name}")
