"""音频加载与预处理"""

import librosa
import numpy as np


def load_audio(file_path: str, sr: int = 22050, mono: bool = True,
               offset: float = 0.0, duration: float = None) -> tuple:
    """
    加载音频文件并进行预处理。

    支持的音频格式（通过 librosa + audioread + soundfile）：
    - WAV  (.wav)
    - MP3  (.mp3)
    - FLAC (.flac)
    - OGG  (.ogg)
    - M4A  (.m4a, .mp4)
    - AAC  (.aac)
    - WMA  (.wma)
    - AIFF (.aiff, .aif)
    - Opus (.opus)
    - AU   (.au)

    Parameters
    ----------
    file_path : str
        音频文件路径（支持上述任意格式）
    sr : int
        目标采样率（Hz），默认 22050
    mono : bool
        是否转为单声道
    offset : float
        起始偏移（秒）
    duration : float
        加载时长（秒），None 表示加载全部

    Returns
    -------
    y : np.ndarray
        音频信号，形状 (n_samples,)
    sr : int
        实际采样率
    """
    y, sr = librosa.load(
        file_path,
        sr=sr,
        mono=mono,
        offset=offset,
        duration=duration
    )
    return y, sr


def preprocess(y: np.ndarray, sr: int) -> np.ndarray:
    """
    音频预处理：去除直流分量、幅度归一化。

    Parameters
    ----------
    y : np.ndarray
        原始音频信号
    sr : int
        采样率

    Returns
    -------
    y_norm : np.ndarray
        归一化后的音频信号
    """
    # 去除直流分量
    y = y - np.mean(y)
    # 幅度归一化到 [-1, 1]
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val
    return y
