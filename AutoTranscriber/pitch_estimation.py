"""多音高估计模块 — 支持和弦识别"""

import numpy as np
from scipy.ndimage import maximum_filter
from collections import defaultdict


def hz_to_midi(freq: float) -> int:
    """将频率 (Hz) 转换为 MIDI 音符号"""
    if freq <= 0:
        return 0
    return int(round(12 * np.log2(freq / 440.0) + 69))


def midi_to_hz(midi_note: int) -> float:
    """将 MIDI 音符号转换为频率 (Hz)"""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


def midi_to_name(midi_note: int) -> str:
    """MIDI 音符号转音名 (如 60 → C4)"""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_note // 12 - 1
    note_name = names[midi_note % 12]
    return f"{note_name}{octave}"


def _is_harmonic_of(candidate_pitch: int, base_pitch: int, max_harmonic: int = 6) -> bool:
    """
    判断 candidate_pitch 是否是 base_pitch 的谐波（八度对齐近似）。

    通过检查 candidate 频率与 base 频率的比值是否为接近整数。
    """
    if candidate_pitch <= base_pitch:
        return False
    # MIDI 音高差 → 频率比
    semitone_diff = candidate_pitch - base_pitch
    freq_ratio = 2 ** (semitone_diff / 12.0)
    # 检查是否接近整数（在 ±0.05 范围内）
    for h in range(2, max_harmonic + 1):
        if abs(freq_ratio - h) < 0.1:
            return True
    return False


def _harmonic_sieve(cqt_frame: np.ndarray, freqs: np.ndarray,
                    n_peaks: int = 6, threshold_factor: float = 0.05,
                    harmonic_tolerance: float = 0.03,
                    max_harmonic: int = 6) -> list:
    """
    迭代谐波减法多音高估计（改进版）。

    专为和弦设计，核心流程：
    1. 在频谱中找到最强峰值作为候选基频
    2. **严格减去**该基频的所有谐波成分能量
    3. 检查候选音符是否为已检测音符的谐波 → 跳过
    4. 重复直到提取足够音符或能量耗尽

    Parameters
    ----------
    cqt_frame : np.ndarray
        当前帧的 CQT 幅度谱，形状 (n_bins,)
    freqs : np.ndarray
        每个 bin 对应的频率 (Hz)
    n_peaks : int
        每帧最大同时音符数
    threshold_factor : float
        幅度阈值
    harmonic_tolerance : float
        谐波匹配容差
    max_harmonic : int
        最多减去多少次谐波

    Returns
    -------
    notes : list of dict
        [{'pitch': int, 'amplitude': float, 'frequency': float}, ...]
    """
    residual = cqt_frame.copy().astype(np.float64)
    original_max = np.max(cqt_frame) if np.max(cqt_frame) > 0 else 1.0
    threshold = threshold_factor * original_max

    detected_notes = []
    min_freq = 65.41
    max_freq = 2093.0

    for iteration in range(n_peaks * 2):  # 多试几次以防中途跳过
        if len(detected_notes) >= n_peaks:
            break

        peak_bin = np.argmax(residual)
        peak_amp = residual[peak_bin]
        peak_freq = freqs[peak_bin]

        if peak_amp < threshold:
            break
        if peak_freq < min_freq or peak_freq > max_freq:
            residual[peak_bin] = 0
            continue

        midi_note = hz_to_midi(peak_freq)
        if midi_note < 12 or midi_note > 127:
            residual[peak_bin] = 0
            continue

        # ---- 检查是否为已检测音符的谐波 ----
        is_harmonic = False
        for prev in detected_notes:
            if _is_harmonic_of(midi_note, prev['pitch'], max_harmonic=6):
                # 是该音符的谐波 → 把这一带能量彻底清空不纳入检测
                bin_range = max(3, int(len(freqs) * 0.008))
                start = max(0, peak_bin - bin_range)
                end = min(len(residual), peak_bin + bin_range)
                residual[start:end] = 0
                is_harmonic = True
                break

        if is_harmonic:
            continue

        # ---- 检查是否与已检测音符同音高（八度重复） ----
        is_duplicate = False
        for prev in detected_notes:
            if midi_note == prev['pitch']:
                is_duplicate = True
                break
        if is_duplicate:
            residual[peak_bin] = 0
            continue

        # ---- 检测到新的音符基频 ----
        detected_notes.append({
            'pitch': midi_note,
            'frequency': float(peak_freq),
            'amplitude': float(peak_amp / original_max)
        })

        # ---- 谐波减法：能量从残差中彻底减去 ----
        for h in range(1, max_harmonic + 1):
            harm_freq = peak_freq * h
            if harm_freq > max_freq * 1.5:
                break

            # 找到谐波附近的 bin
            freq_diff = np.abs(freqs / harm_freq - 1.0)
            match_idx = np.where(freq_diff < harmonic_tolerance)[0]

            for idx in match_idx:
                # 根据检测到的幅度，按比例减去
                subtraction = peak_amp * min(1.0, (0.9 / h))
                residual[idx] = max(0, residual[idx] - subtraction)

        # 清空峰值周围
        bin_range = max(3, int(len(freqs) * 0.008))
        start = max(0, peak_bin - bin_range)
        end = min(len(residual), peak_bin + bin_range)
        residual[start:end] = 0

    # 按幅度排序
    detected_notes.sort(key=lambda n: n['amplitude'], reverse=True)
    return detected_notes


def estimate_vocal_pitch(y: np.ndarray, sr: int,
                          hop_length: int = 512,
                          fmin: float = 65.41,
                          fmax: float = 2093.0,
                          min_duration_frames: int = 3) -> list:
    """
    使用 pYIN 算法对人声进行高精度单音音高估计。

    pYIN 是 YIN 算法的概率改进版，专门为单音音高估计设计，
    对人声的颤音、滑音等效果远好于 CQT + 峰值检测。

    Parameters
    ----------
    y : np.ndarray
        音频信号（人声轨）
    sr : int
        采样率
    hop_length : int
        帧移
    fmin : float
        最低频率 (Hz)
    fmax : float
        最高频率 (Hz)
    min_duration_frames : int
        最小音符持续帧数（过滤短噪声）

    Returns
    -------
    notes : list of dict
        [{'start': float, 'end': float, 'pitch': int, 'velocity': int}, ...]
    """
    import librosa

    # pYIN 音高估计
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        fill_na=0.0
    )

    # 将频率转为 MIDI 音符号
    # 只保留 voiced 且概率 > 0.3 的帧
    threshold = 0.3
    midi_notes = np.zeros_like(f0, dtype=int)
    for i, (f, v) in enumerate(zip(f0, voiced_flag)):
        if v and f > 0 and voiced_prob[i] > threshold:
            midi = int(round(12 * np.log2(f / 440.0) + 69))
            if 0 < midi < 128:
                midi_notes[i] = midi
        else:
            midi_notes[i] = 0

    # ---- 将连续帧合并为音符事件 ----
    times = librosa.frames_to_time(
        np.arange(len(midi_notes)),
        sr=sr, hop_length=hop_length
    )

    notes = []
    cur_pitch = 0
    cur_start = 0
    cur_count = 0

    for i, pitch in enumerate(midi_notes):
        if pitch > 0:
            if cur_pitch == 0:
                # 新音符开始
                cur_pitch = pitch
                cur_start = i
                cur_count = 1
            elif pitch == cur_pitch:
                # 同音高延续
                cur_count += 1
            else:
                # 音高变化：提交旧音符
                if cur_count >= min_duration_frames:
                    notes.append({
                        'start': float(times[cur_start]),
                        'end': float(times[i]),
                        'pitch': int(cur_pitch),
                        'velocity': 80
                    })
                # 开始新音符
                cur_pitch = pitch
                cur_start = i
                cur_count = 1
        else:
            if cur_pitch > 0:
                # 静音：提交当前音符
                if cur_count >= min_duration_frames:
                    notes.append({
                        'start': float(times[cur_start]),
                        'end': float(times[i]),
                        'pitch': int(cur_pitch),
                        'velocity': 80
                    })
                cur_pitch = 0
                cur_count = 0

    # 处理最后一个音符
    if cur_pitch > 0 and cur_count >= min_duration_frames:
        notes.append({
            'start': float(times[cur_start]),
            'end': float(times[-1]),
            'pitch': int(cur_pitch),
            'velocity': 80
        })

    return notes


def estimate_pitches(cqt: np.ndarray, freqs: np.ndarray,
                     times: np.ndarray, sr: int,
                     hop_length: int = 512,
                     n_peaks: int = 6,
                     min_freq: float = 65.41,
                     max_freq: float = 2093.0,
                     threshold_factor: float = 0.05,
                     use_harmonic_sieve: bool = True) -> list:
    """
    对 CQT 谱的每个时间帧进行多音高估计。

    Parameters
    ----------
    use_harmonic_sieve : bool
        True → 谐波减法（适合和弦，推荐）
        False → 传统峰值检测（适合单音旋律，速度快）

    Returns
    -------
    frame_notes : list of list of dict
    """
    n_frames = cqt.shape[1]
    frame_notes = []

    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)

    for t in range(n_frames):
        frame = cqt[:, t].copy()
        frame[~freq_mask] = 0

        if np.max(frame) == 0:
            frame_notes.append([])
            continue

        if use_harmonic_sieve:
            notes = _harmonic_sieve(
                frame, freqs,
                n_peaks=n_peaks,
                threshold_factor=threshold_factor
            )
        else:
            notes = _peak_picking(
                frame, freqs,
                n_peaks=n_peaks,
                threshold_factor=threshold_factor
            )

        frame_notes.append(notes)

    return frame_notes


def _peak_picking(cqt_frame: np.ndarray, freqs: np.ndarray,
                  n_peaks: int = 4, threshold_factor: float = 0.1) -> list:
    """传统峰值检测（备用）"""
    frame_norm = cqt_frame / np.max(cqt_frame) if np.max(cqt_frame) > 0 else cqt_frame
    frame_max = maximum_filter(frame_norm, size=5, mode='constant')
    is_peak = (frame_norm == frame_max) & (frame_norm > threshold_factor)
    peak_indices = np.where(is_peak)[0]
    peak_amps = cqt_frame[peak_indices]
    sorted_order = np.argsort(peak_amps)[::-1]
    peak_indices = peak_indices[sorted_order][:n_peaks]
    peak_amps = peak_amps[sorted_order][:n_peaks]
    notes = []
    for idx, amp in zip(peak_indices, peak_amps):
        freq = freqs[idx]
        midi_note = hz_to_midi(freq)
        if 0 < midi_note < 128:
            notes.append({
                'pitch': midi_note,
                'frequency': float(freq),
                'amplitude': float(amp / np.max(cqt_frame))
            })
    return notes
