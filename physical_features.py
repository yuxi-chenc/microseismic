# -*- coding: utf-8 -*-
"""
Physical Features Calculator for Seismic/Microseismic Signals
包含了基于滑动窗口的 20+ 个物理属性特征计算函数。
整合了基础统计、波形形态 (No.2-8)、自相关 (No.9-23) 及频域特征 (No.24-40)。
"""

import numpy as np
from scipy.signal import hilbert, find_peaks

# ==========================================
# 0. 基础工具函数 (Utils)
# ==========================================

def min_max_normalize(data_array):
    """归一化工具：将特征缩放到 0-1 之间"""
    data_min = np.min(data_array)
    data_max = np.max(data_array)
    if data_max - data_min == 0:
        return np.zeros_like(data_array)
    return (data_array - data_min) / (data_max - data_min)


def normalized_mean_removal(data_array):
    data_min = np.min(data_array)
    data_max = np.max(data_array)
    
    if data_max - data_min == 0:
        return np.zeros_like(data_array)
    
    norm_data = (data_array - data_min) / (data_max - data_min)
    
    return norm_data - np.mean(norm_data)


def get_slice(data, i, window, data_length):
    """滑动窗口切片工具"""
    start = max(0, int(i - window / 2))
    end = min(data_length, int(i + window / 2))
    return data[start:end]

# ==========================================
# 1. 基础能量与统计特征 (Basic Stats)
# ==========================================

def get_sum_sq_diff(data, window, data_length):
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 1:
            results.append(0)
            continue
        mean_val = np.mean(slice_data)
        val = np.sum((slice_data - mean_val)**2)
        results.append(val)
    return min_max_normalize(np.array(results))


def get_rms(data, window, data_length):
    """计算均方根 (RMS) - 能量指标"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 1:
            results.append(0)
            continue
        val = np.sqrt(np.mean(slice_data**2))
        results.append(val)
    return min_max_normalize(np.array(results))

def get_std(data, window, data_length):
    """计算标准差 (STD/G) - 波动幅度"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 1:
            results.append(0)
            continue
        val = np.std(slice_data)
        results.append(val)
    return min_max_normalize(np.array(results))

def get_energy(data, window, data_length):
    """计算绝对振幅和 - 简化版能量"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        val = np.sum(np.abs(slice_data))
        results.append(val)
    return min_max_normalize(np.array(results))

def get_zcr(data, window, data_length):
    """计算过零率 (Zero Crossing Rate) - 粗略频率"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        val = ((slice_data[:-1] * slice_data[1:]) < 0).sum() / len(slice_data)
        results.append(val)
    return min_max_normalize(np.array(results))

# ==========================================
# 2. 波形形态特征 (No. 2 - No. 8)
# ==========================================

def get_env_mean_max_ratio(data, window, data_length):
    """[No. 2] 包络均值比: 描述信号饱满度"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        envelope = np.abs(hilbert(slice_data))
        if np.max(envelope) == 0:
            results.append(0)
        else:
            results.append(np.mean(envelope) / np.max(envelope))
    return min_max_normalize(np.array(results))

def get_env_median_max_ratio(data, window, data_length):
    """[No. 3] 包络中值比"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        envelope = np.abs(hilbert(slice_data))
        if np.max(envelope) == 0:
            results.append(0)
        else:
            results.append(np.median(envelope) / np.max(envelope))
    return min_max_normalize(np.array(results))

def get_rise_fall_ratio(data, window, data_length):
    """[No. 4] 上升下降时间比: 区分机器(≈1)与落石(<1)"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        length = len(slice_data)
        if length < 2:
            results.append(0)
            continue
        idx_max = np.argmax(np.abs(slice_data))
        t_asc = idx_max
        t_desc = length - 1 - idx_max
        if t_desc == 0:
            results.append(10.0)
        else:
            results.append(t_asc / t_desc)
    return min_max_normalize(np.array(results))

def get_raw_kurtosis(data, window, data_length):
    """[No. 5] 原始信号峰度 (K): 描述突发性"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        mean = np.mean(slice_data)
        std = np.std(slice_data)
        if std == 0:
            results.append(0)
        else:
            val = np.mean((slice_data - mean) ** 4) / (std ** 4)
            results.append(val)
    return min_max_normalize(np.array(results))

def get_env_kurtosis(data, window, data_length):
    """[No. 6] 包络峰度"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        envelope = np.abs(hilbert(slice_data))
        mean = np.mean(envelope)
        std = np.std(envelope)
        if std == 0:
            results.append(0)
        else:
            val = np.mean((envelope - mean) ** 4) / (std ** 4)
            results.append(val)
    return min_max_normalize(np.array(results))

def get_raw_skewness(data, window, data_length):
    """[No. 7] 原始信号偏度 (S): 描述不对称性"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        mean = np.mean(slice_data)
        std = np.std(slice_data)
        if std == 0:
            results.append(0)
        else:
            val = abs(np.mean((slice_data - mean) ** 3) / (std ** 3))
            results.append(val)
    return min_max_normalize(np.array(results))

def get_env_skewness(data, window, data_length):
    """[No. 8] 包络偏度"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        envelope = np.abs(hilbert(slice_data))
        mean = np.mean(envelope)
        std = np.std(envelope)
        if std == 0:
            results.append(0)
        else:
            val = abs(np.mean((envelope - mean) ** 3) / (std ** 3))
            results.append(val)
    return min_max_normalize(np.array(results))

def get_crest_factor(data, window, data_length):
    """[Bonus] 波峰因数 (Peak/RMS): 极佳的撞击检测特征"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 1:
            results.append(0)
            continue
        abs_max = np.max(np.abs(slice_data))
        rms = np.sqrt(np.mean(slice_data**2))
        val = abs_max / (rms + 1e-9)
        results.append(val)
    return min_max_normalize(np.array(results))

# ==========================================
# 3. 自相关与衰减特征 (No. 9 - No. 23)
# ==========================================

def get_autocorr_peaks(data, window, data_length):
    """[No. 9] 自相关峰值数: 区分机械/自然信号"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 10:
            results.append(0)
            continue
        centered = slice_data - np.mean(slice_data)
        if np.std(centered) == 0:
            results.append(0)
            continue
        acf = np.correlate(centered, centered, mode='full')
        acf = acf[len(acf)//2:] 
        peaks, _ = find_peaks(acf, height=acf[0]*0.1, distance=5)
        results.append(np.sum(peaks > 0))
    return min_max_normalize(np.array(results))

def get_autocorr_energy_ratio(data, window, data_length):
    """[No. 12] 自相关能量比 (后部/前部)"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 10:
            results.append(0)
            continue
        centered = slice_data - np.mean(slice_data)
        acf = np.correlate(centered, centered, mode='full')
        acf = acf[len(acf)//2:]
        split_idx = len(acf) // 3
        energy_first = np.sum(np.abs(acf[:split_idx]))
        energy_rest = np.sum(np.abs(acf[split_idx:]))
        if energy_first == 0:
            results.append(0)
        else:
            results.append(energy_rest / energy_first)
    return min_max_normalize(np.array(results))

def get_linear_decay_error(data, window, data_length):
    """[No. 23] 线性衰减拟合误差"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 5:
            results.append(0)
            continue
        envelope = np.abs(hilbert(slice_data))
        idx_max = np.argmax(envelope)
        decay_part = envelope[idx_max:]
        if len(decay_part) < 2:
            results.append(0)
            continue
        ideal_line = np.linspace(decay_part[0], decay_part[-1], len(decay_part))
        rms_error = np.sqrt(np.mean((decay_part - ideal_line) ** 2))
        results.append(rms_error)
    return min_max_normalize(np.array(results))

# ==========================================
# 4. 频谱特征 (No. 24 - No. 40)
# ==========================================

def get_spec_mean(data, window, data_length):
    """[No. 24] 频谱均值"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        mag = np.abs(np.fft.rfft(slice_data))
        results.append(np.mean(mag))
    return min_max_normalize(np.array(results))

def get_dom_freq(data, window, data_length, fs=100.0):
    """[No. 26] 主频 (Max Frequency)"""
    results = []
    freqs = np.fft.rfftfreq(window, d=1/fs)
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        cur_len = len(slice_data)
        if cur_len < 2:
            results.append(0)
            continue
        if cur_len == window:
            cur_freqs = freqs
        else:
            cur_freqs = np.fft.rfftfreq(cur_len, d=1/fs)
        mag = np.abs(np.fft.rfft(slice_data))
        results.append(cur_freqs[np.argmax(mag)])
    return min_max_normalize(np.array(results))

def get_quartile_freq(data, window, data_length, fs=100.0):
    """[No. 28] 频率中位数 (2nd Quartile Freq)"""
    results = []
    freqs = np.fft.rfftfreq(window, d=1/fs)
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        cur_freqs = freqs if len(slice_data) == window else np.fft.rfftfreq(len(slice_data), d=1/fs)
        mag = np.abs(np.fft.rfft(slice_data))
        cumsum = np.cumsum(mag)
        if cumsum[-1] == 0:
            results.append(0)
            continue
        idx = np.searchsorted(cumsum, cumsum[-1] * 0.5)
        results.append(cur_freqs[min(idx, len(cur_freqs)-1)])
    return min_max_normalize(np.array(results))

def get_spec_peaks_count(data, window, data_length):
    """[No. 31] 显著频谱峰值数 (>0.75 Max)"""
    results = []
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        mag = np.abs(np.fft.rfft(slice_data))
        mx = np.max(mag)
        if mx == 0:
            results.append(0)
        else:
            peaks, _ = find_peaks(mag, height=mx * 0.75)
            results.append(len(peaks))
    return min_max_normalize(np.array(results))

def get_nyquist_band_energy(data, window, data_length, fs=100.0):
    """[No. 34] 低频段能量 (0 - 1/4 Nyquist)"""
    results = []
    limit = (fs / 2.0) * 0.25
    freqs = np.fft.rfftfreq(window, d=1/fs)
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        cur_freqs = freqs if len(slice_data) == window else np.fft.rfftfreq(len(slice_data), d=1/fs)
        mag = np.abs(np.fft.rfft(slice_data))
        results.append(np.sum(mag[cur_freqs <= limit]))
    return min_max_normalize(np.array(results))

def get_spec_centroid(data, window, data_length, fs=100.0):
    """[No. 38] 频谱质心 (Centroid)"""
    results = []
    freqs = np.fft.rfftfreq(window, d=1/fs)
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        cur_freqs = freqs if len(slice_data) == window else np.fft.rfftfreq(len(slice_data), d=1/fs)
        mag = np.abs(np.fft.rfft(slice_data))
        sm = np.sum(mag)
        if sm == 0:
            results.append(0)
        else:
            results.append(np.sum(cur_freqs * mag) / sm)
    return min_max_normalize(np.array(results))

def get_gyration_radius(data, window, data_length, fs=100.0):
    """[No. 39] 回转半径 (Gyration Radius)"""
    results = []
    freqs = np.fft.rfftfreq(window, d=1/fs)
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        cur_freqs = freqs if len(slice_data) == window else np.fft.rfftfreq(len(slice_data), d=1/fs)
        mag = np.abs(np.fft.rfft(slice_data))
        sm = np.sum(mag)
        if sm == 0:
            results.append(0)
        else:
            moment_2 = np.sum((cur_freqs ** 2) * mag)
            results.append(np.sqrt(moment_2 / sm))
    return min_max_normalize(np.array(results))

def get_spec_bandwidth(data, window, data_length, fs=100.0):
    """[No. 40] 频谱带宽 (Spectral Width)"""
    results = []
    freqs = np.fft.rfftfreq(window, d=1/fs)
    for i in range(data_length):
        slice_data = get_slice(data, i, window, data_length)
        if len(slice_data) < 2:
            results.append(0)
            continue
        cur_freqs = freqs if len(slice_data) == window else np.fft.rfftfreq(len(slice_data), d=1/fs)
        mag = np.abs(np.fft.rfft(slice_data))
        sm = np.sum(mag)
        if sm == 0:
            results.append(0)
        else:
            centroid = np.sum(cur_freqs * mag) / sm
            results.append(np.sqrt(np.sum(((cur_freqs - centroid) ** 2) * (mag / sm))))
    return min_max_normalize(np.array(results))