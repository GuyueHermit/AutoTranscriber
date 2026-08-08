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

    # ---- librosa 风格局部 max 归一化 ----
    # 长音/持续段在局部窗口内接近其包络峰值 → norm ~ 1；
    # 瞬态突跃（真正的 onset）相对自身包络也接近 1，但持续段不会
    # 因为绝对能量高而压过弱 onset。归一化消除了动态范围的影响。
    window = int(0.1 * sr / hop_length)  # 约 100ms 的窗口
    if window < 3:
        window = 3
    from scipy.ndimage import maximum_filter1d
    local_env = maximum_filter1d(flux_smooth, size=2 * window + 1, mode='nearest')
    norm = flux_smooth / (local_env + 1e-12)

    # 绝对能量门限：防止纯归一化在安静段把噪声当 onset
    energy_floor = 0.05 * np.max(flux_smooth)

    # onset 得分 = 归一化突跃，但绝对能量必须达标
    onset_score = np.where(flux_smooth >= energy_floor, norm, 0.0)

    # 峰值检测（阈值相对归一化值：0.5 ≈ 能量翻倍级突跃）
    peaks, _ = find_peaks(
        onset_score,
        height=threshold * 0.6,
        distance=min_distance
    )

    # 兜底：局部均值自适应阈值（原逻辑，更宽松的 offset）
    if len(peaks) == 0:
        local_mean = np.convolve(
            flux_smooth,
            np.ones(window) / window,
            mode='same'
        )
        adaptive_threshold = local_mean + threshold * 0.3 * np.max(flux_smooth)
        flux_diff = np.maximum(flux_smooth - adaptive_threshold, 0)
        peaks, _ = find_peaks(
            flux_diff,
            height=1e-9,
            distance=min_distance
        )
        if len(peaks) == 0:  # 最后兜底：绝对阈值
            peaks, _ = find_peaks(
                flux_smooth,
                height=threshold * 0.1 * np.max(flux_smooth),
                distance=min_distance
            )

    # 算时间
    onset_frames = peaks
    onset_times = onset_frames * hop_length / sr

    return onset_frames, onset_times
