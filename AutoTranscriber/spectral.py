"""频谱分析模块 — CQT 变换与频谱通量计算"""

import librosa
import numpy as np
from scipy.ndimage import maximum_filter


def compute_cqt(y: np.ndarray, sr: int, hop_length: int = 512,
                fmin: float = 65.41, fmax: float = 2093.0,
                bins_per_octave: int = 36) -> tuple:
    """
    计算 Constant-Q Transform (CQT) 频谱。
    CQT 的频率轴为对数刻度，更符合人耳听觉和音乐感知。

    Parameters
    ----------
    y : np.ndarray
        音频信号
    sr : int
        采样率
    hop_length : int
        帧移（样本点数）
    fmin : float
        最低频率 (Hz)，默认 65.41 Hz (C2)
    fmax : float
        最高频率 (Hz)，默认 2093.0 Hz (C7)
    bins_per_octave : int
        每八度的频带数，默认 36（即每 3 个 bin 对应一个半音）,
        设为 12 则每个 bin 对应一个半音

    Returns
    -------
    cqt : np.ndarray
        CQT 幅度谱，形状 (n_bins, n_frames)
    times : np.ndarray
        每帧对应的时间点（秒）
    freqs : np.ndarray
        每个 bin 对应的频率 (Hz)
    """
    n_bins = int(np.ceil(bins_per_octave * np.log2(fmax / fmin)))

    cqt = np.abs(librosa.cqt(
        y,
        sr=sr,
        hop_length=hop_length,
        fmin=fmin,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        tuning=0.0,
        filter_scale=1,
        norm=1,
        window='hann',
        pad_mode='reflect'
    ))

    times = librosa.frames_to_time(
        np.arange(cqt.shape[1]),
        sr=sr,
        hop_length=hop_length
    )
    freqs = librosa.cqt_frequencies(
        n_bins=n_bins,
        fmin=fmin,
        bins_per_octave=bins_per_octave
    )

    return cqt, times, freqs


def compute_spectral_flux(cqt: np.ndarray) -> np.ndarray:
    """
    计算频谱通量（Spectral Flux）。
    即 CQT 频谱沿时间轴的幅度增量之和，
    用于检测音符起始位置。

    Parameters
    ----------
    cqt : np.ndarray
        CQT 幅度谱，形状 (n_bins, n_frames)

    Returns
    -------
    flux : np.ndarray
        频谱通量，形状 (n_frames,)
    """
    # 对数幅度，提升弱信号的可见性
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)

    # 半波整流差分：只取正变化（能量增加）
    diff = np.diff(cqt_db, axis=1)
    diff = np.maximum(diff, 0)

    # 沿频率轴求和
    flux = np.sum(diff, axis=0)

    # 归一化
    if np.max(flux) > 0:
        flux = flux / np.max(flux)

    return flux


def compute_stft(y: np.ndarray, sr: int, hop_length: int = 512,
                 n_fft: int = 2048) -> tuple:
    """
    计算短时傅里叶变换 (STFT) 频谱（备选）。

    Parameters
    ----------
    y : np.ndarray
        音频信号
    sr : int
        采样率
    hop_length : int
        帧移
    n_fft : int
        FFT 窗口大小

    Returns
    -------
    stft : np.ndarray
        STFT 幅度谱
    times : np.ndarray
        时间点（秒）
    freqs : np.ndarray
        频率轴 (Hz)
    """
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    times = librosa.frames_to_time(
        np.arange(D.shape[1]),
        sr=sr,
        hop_length=hop_length
    )
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return D, times, freqs
