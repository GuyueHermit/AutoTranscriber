"""音符起始检测模块"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def detect_onsets(flux: np.ndarray, sr: int,
                  hop_length: int = 512,
                  threshold: float = 0.5,
                  min_distance: int = 3,
                  smooth_sigma: float = 1.0) -> np.ndarray:
    """
    基于频谱通量的音符起始检测。

    流程：
    1. 对频谱通量进行高斯平滑
    2. 使用自适应阈值（局部均值 + offset）提取峰值
    3. 返回检测到的起始帧索引

    Parameters
    ----------
    flux : np.ndarray
        频谱通量，形状 (n_frames,)
    sr : int
        采样率
    hop_length : int
        帧移
    threshold : float
        峰值检测的相对阈值（相对于通量最大值）
    min_distance : int
        两个起始点之间的最小帧数间隔
    smooth_sigma : float
        高斯平滑的 sigma 值（帧数单位）

    Returns
    -------
    onset_frames : np.ndarray
        起始帧索引
    onset_times : np.ndarray
        起始时间（秒）
    """
    # 高斯平滑
    flux_smooth = gaussian_filter1d(flux, sigma=smooth_sigma)

    # 自适应阈值: 局部均值 + offset
    window = int(0.1 * sr / hop_length)  # 约 100ms 的窗口
    if window < 3:
        window = 3
    local_mean = np.convolve(
        flux_smooth,
        np.ones(window) / window,
        mode='same'
    )
    adaptive_threshold = local_mean + threshold * np.max(flux_smooth)

    # 计算原始通量减去自适应阈值，找到正峰值
    flux_diff = flux_smooth - adaptive_threshold
    flux_diff = np.maximum(flux_diff, 0)  # 半波整流

    # 找峰值
    peaks, properties = find_peaks(
        flux_diff,
        height=threshold * 0.1 * np.max(flux_diff),
        distance=min_distance
    )

    # 如果没有检测到峰值，尝试降低门槛
    if len(peaks) == 0:
        peaks, properties = find_peaks(
            flux_diff,
            height=0.01,
            distance=min_distance
        )

    # 算时间
    onset_frames = peaks
    onset_times = onset_frames * hop_length / sr

    return onset_frames, onset_times
