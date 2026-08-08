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


def _is_harmonic_of_freq(candidate_freq: float, base_freq: float,
                         max_harmonic: int = 6,
                         tolerance: float = 0.06) -> bool:
    """
    用连续频率比判断 candidate_freq 是否为 base_freq 的整数倍谐波。

    相比旧版（MIDI 音高差 -> 2^(n/12) 近似频率比），直接用真实频率比值，
    避免半音量化误差导致的误判/漏判。容差为绝对容差（~2 个 CQT bin 定位误差）。
    """
    if base_freq <= 0 or candidate_freq <= base_freq * 1.02:
        return False
    ratio = candidate_freq / base_freq
    h = int(round(ratio))
    if h < 2 or h > max_harmonic:
        return False
    return abs(ratio - h) <= tolerance


def _is_harmonic_of(candidate_pitch: int, base_pitch: int, max_harmonic: int = 6) -> bool:
    """
    判断 candidate_pitch 是否是 base_pitch 的谐波（连续频率比版本）。

    通过检查 candidate 频率与 base 频率的比值是否为接近整数。
    """
    if candidate_pitch <= base_pitch:
        return False
    return _is_harmonic_of_freq(
        midi_to_hz(candidate_pitch),
        midi_to_hz(base_pitch),
        max_harmonic=max_harmonic
    )


def build_harmonic_table(freqs: np.ndarray, max_harmonic: int = 6,
                         harmonic_tolerance: float = 0.03) -> list:
    """
    预计算谐波索引表：对每个 bin（作为潜在基频），给出其 1..max_harmonic 次
    谐波命中的所有 bin 索引及先验权重（1/h，谐波能量随次数递减）。

    Parameters
    ----------
    freqs : np.ndarray
        每个 CQT bin 对应的频率 (Hz)
    max_harmonic : int
        最多考虑几次谐波
    harmonic_tolerance : float
        谐波匹配容差（频率相对误差）

    Returns
    -------
    table : list of (indices, weights)
        table[b] = (harmonic_bin_indices, harmonic_bin_weights)，
        帧循环内直接查表，避免逐帧重复 `freqs / harm_freq` 全数组比较。
    """
    n_bins = len(freqs)
    table = []
    for b in range(n_bins):
        f0 = freqs[b]
        idx_map = {}  # bin -> 权重（多个谐波命中同一 bin 时取最大权重）
        for h in range(1, max_harmonic + 1):
            harm_freq = f0 * h
            freq_diff = np.abs(freqs / harm_freq - 1.0)
            match_idx = np.where(freq_diff < harmonic_tolerance)[0]
            w = 1.0 / h
            for m in match_idx:
                if w > idx_map.get(int(m), 0.0):
                    idx_map[int(m)] = w
        indices = np.array(sorted(idx_map.keys()), dtype=int)
        weights = np.array([idx_map[i] for i in indices], dtype=float)
        table.append((indices, weights))
    return table


def _harmonic_sieve(cqt_frame: np.ndarray, freqs: np.ndarray,
                    n_peaks: int = 6, threshold_factor: float = 0.05,
                    harmonic_tolerance: float = 0.03,
                    max_harmonic: int = 6,
                    harmonic_table: list = None,
                    top_candidates: int = 16,
                    suppress_ratio: float = None) -> list:
    """
    谐波模板 + NNLS 非负最小二乘多音高估计（改进版）。

    相比旧版「迭代谐波减法」（固定 0.9/h 比例减去、逐帧 argmax 贪心），
    本版本：
    1. **并行候选评估** — 每帧取 Top-K 局部峰值作为候选基频，同时求解，
       不再贪心先到先得（两个共享谐波的基频会分摊能量而非互相压制）
    2. **NNLS 模板拟合** — 为每个候选建立谐波模板（1/h 能量先验），
       用非负最小二乘同时估计所有候选强度，适配真实乐器谐波分布
    3. **预计算查表** — harmonic_table 一次构建，帧循环内直接查表，
       不再逐帧重复 `freqs / harm_freq` 全数组比较
    4. **连续频率比谐波抑制** — 用真实频率比值判断整数倍关系

    Parameters
    ----------
    cqt_frame : np.ndarray
        当前帧的 CQT 幅度谱，形状 (n_bins,)
    freqs : np.ndarray
        每个 bin 对应的频率 (Hz)
    n_peaks : int
        每帧最大同时音符数
    threshold_factor : float
        强度阈值（相对 NNLS 最大拟合强度）
    harmonic_tolerance : float
        谐波匹配容差
    max_harmonic : int
        最多考虑多少次谐波
    harmonic_table : list, optional
        build_harmonic_table 的预计算结果；None 则内部惰性构建
    top_candidates : int
        每帧局部峰值候选数（参与 NNLS 求解的上限）
    suppress_ratio : float, optional
        显式谐波抑制比例（默认 None = 信任 NNLS 能量分摊，不做显式抑制；
        NNLS 模板拟合已天然把基频的泛音能量归入基频，谐波候选的剩余
        系数即其独立音符证据）。若传入数值，则仅当低音候选强度 >=
        suppress_ratio * 高音候选强度时才抑制（1.0 = 无条件抑制，
        相当于旧版行为）。

    Returns
    -------
    notes : list of dict
        [{'pitch': int, 'amplitude': float, 'frequency': float}, ...]
    """
    from scipy.optimize import nnls

    frame = cqt_frame.astype(np.float64)
    original_max = np.max(frame) if np.max(frame) > 0 else 1.0

    if harmonic_table is None:
        harmonic_table = build_harmonic_table(
            freqs, max_harmonic, harmonic_tolerance)

    # ---- 1. 候选基频：局部峰值 Top-K（并行评估，非贪心） ----
    kernel = maximum_filter(frame, size=5, mode='constant')
    is_peak = (frame == kernel) & (frame > 0)
    peak_bins = np.where(is_peak)[0]
    if len(peak_bins) == 0:
        return []

    peak_amps = frame[peak_bins]
    # 过滤基频 bin 自身能量过低的峰值：避免 NNLS 的 missing-fundamental
    # 效应（低频幽灵候选通过高次谐波解释其他基频的能量）
    min_peak = threshold_factor * original_max
    valid = peak_amps >= min_peak
    peak_bins = peak_bins[valid]
    peak_amps = peak_amps[valid]
    if len(peak_bins) == 0:
        return []

    top_idx = np.argsort(peak_amps)[::-1][:top_candidates]
    candidates = peak_bins[top_idx]

    # ---- 2. NNLS 模板拟合 ----
    n_bins = len(frame)
    n_cand = len(candidates)
    template = np.zeros((n_bins, n_cand))
    for j, b in enumerate(candidates):
        idxs, wts = harmonic_table[b]
        template[idxs, j] = wts

    # 丢弃模板为空的候选（谐波全落在频率范围外）
    active = np.where(template.max(axis=0) > 0)[0]
    if len(active) == 0:
        return []
    template = template[:, active]
    candidates = candidates[active]

    strengths, _ = nnls(template, frame)

    # ---- 3. 谐波抑制（默认 None = 信任 NNLS 分摊；显式抑制仅作可选开关） ----
    min_freq = 65.41
    max_freq = 2093.0
    kept = []  # (bin, freq, strength)

    order = np.argsort(strengths)[::-1]
    for j in order:
        s = strengths[j]
        if s <= 0:
            continue
        freq_c = freqs[candidates[j]]
        if freq_c < min_freq or freq_c > max_freq:
            continue

        if suppress_ratio is not None:
            is_harm = False
            for (_, freq_k, s_k) in kept:
                if _is_harmonic_of_freq(freq_c, freq_k, max_harmonic):
                    if s_k >= suppress_ratio * s:
                        is_harm = True
                        break
            if is_harm:
                continue

        kept.append((int(candidates[j]), float(freq_c), float(s)))
        if len(kept) >= n_peaks:
            break

    # ---- 4. 阈值过滤 + 输出 ----
    max_strength = max(strengths) if len(strengths) > 0 else 0.0
    threshold = threshold_factor * max_strength if max_strength > 0 else 0.0

    detected_notes = []
    for b, f, s in kept:
        if s < threshold:
            continue
        midi_note = hz_to_midi(f)
        if midi_note < 12 or midi_note > 127:
            continue
        detected_notes.append({
            'pitch': midi_note,
            'frequency': f,
            'amplitude': float(s / original_max)
        })

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
    from scipy.signal import medfilt

    # pYIN 音高估计
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        hop_length=hop_length,
        fill_na=0.0
    )

    # ---- 概率阈值 + 中值滤波平滑（HMM 平滑的轻量替代） ----
    # 1) voiced 状态序列中值滤波：消除单帧孤立跳变（颤音/呼吸噪声）
    # 2) f0 序列中值滤波：平滑抖动的音高轨迹
    threshold = 0.3
    voiced = (voiced_flag > 0) & (voiced_prob > threshold) & (f0 > 0)
    voiced = medfilt(voiced.astype(int), kernel_size=5).astype(bool)

    f0_smooth = f0.copy()
    voiced_idx = np.where(voiced)[0]
    if len(voiced_idx) > 0:
        f0_smooth[voiced_idx] = medfilt(f0[voiced_idx], kernel_size=5)

    # 将频率转为 MIDI 音符号
    midi_notes = np.zeros_like(f0, dtype=int)
    for i in range(len(f0)):
        if voiced[i]:
            midi = int(round(12 * np.log2(f0_smooth[i] / 440.0) + 69))
            if 0 < midi < 128:
                midi_notes[i] = midi

    # ---- 帧能量包络（用于动态力度映射） ----
    frame_rms = librosa.feature.rms(y=y, frame_length=2048,
                                    hop_length=hop_length)[0]
    if frame_rms.size < len(midi_notes):
        frame_rms = np.pad(frame_rms, (0, len(midi_notes) - frame_rms.size))
    rms_ref = float(np.percentile(frame_rms[frame_rms > 0], 95)) \
        if np.any(frame_rms > 0) else 1.0
    rms_ref = max(rms_ref, 1e-8)

    def _vel_from_energy(start_idx: int, end_idx: int) -> int:
        """将音符段平均能量映射到 MIDI 力度 (1~127)"""
        seg = frame_rms[start_idx:end_idx]
        e = float(np.mean(seg)) if len(seg) > 0 else 0.0
        return min(127, max(1, int(40 + 80 * min(1.0, e / rms_ref))))

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
                        'velocity': _vel_from_energy(cur_start, i)
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
                        'velocity': _vel_from_energy(cur_start, i)
                    })
                cur_pitch = 0
                cur_count = 0

    # 处理最后一个音符
    if cur_pitch > 0 and cur_count >= min_duration_frames:
        notes.append({
            'start': float(times[cur_start]),
            'end': float(times[-1]),
            'pitch': int(cur_pitch),
            'velocity': _vel_from_energy(cur_start, len(midi_notes))
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

    # 预计算谐波索引表一次，帧循环内查表（大幅减少重复频率比较）
    harmonic_table = None
    if use_harmonic_sieve:
        harmonic_table = build_harmonic_table(freqs)

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
                threshold_factor=threshold_factor,
                harmonic_table=harmonic_table
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



# ============================================================================
#  新方法：基于起始点的音高检测
#  只检测每个起始点「新进入」的音，解决余音干扰问题
# ============================================================================

def _spectral_delta_filter(cqt: np.ndarray, onset_frames: np.ndarray,
                           lookback: int = 3) -> np.ndarray:
    """
    对每个起始点，计算其与之前安静帧的频谱差（delta）。
    只返回「新出现」的能量，过滤持续存在的余音。

    Parameters
    ----------
    cqt : np.ndarray
        CQT 频谱，形状 (n_bins, n_frames)
    onset_frames : np.ndarray
        起始帧索引
    lookback : int
        取起始前多少帧的均值作为基准（参考安静段）

    Returns
    -------
    delta_cqt : np.ndarray
        与 cqt 同形的 delta 频谱，只有新进入的能量非零
    """
    delta_cqt = np.zeros_like(cqt)
    n_frames = cqt.shape[1]

    for onset in onset_frames:
        if onset < lookback or onset >= n_frames:
            continue

        # 取起始前 lookback 帧的平均作为基准（安静段）
        base = np.mean(cqt[:, onset - lookback:onset], axis=1)
        # 起始时刻的频谱
        current = cqt[:, onset]
        # delta = 新出现的能量
        delta = np.maximum(0, current - base * 1.2)  # 1.2 倍容忍
        delta_cqt[:, onset] = delta

    return delta_cqt


def estimate_pitches_onset_driven(
    cqt: np.ndarray, freqs: np.ndarray, times: np.ndarray,
    onset_frames: np.ndarray, onset_times: np.ndarray,
    sr: int, hop_length: int = 512,
    n_peaks: int = 5,
    min_freq: float = 65.41,
    max_freq: float = 2093.0,
    threshold_factor: float = 0.08
) -> list:
    """
    基于起始点的音高检测——只在每个起始点检测「新进入」的音。

    与旧方法的核心区别：
    1. 不是逐帧检测，而是只在起始点检测
    2. 用频谱差（delta）替代原始频谱，排除旧音的余响
    3. 返回结构也包含 onset_frame 信息，方便后续追踪持续时间

    Parameters
    ----------
    cqt : np.ndarray
        CQT 频谱 (n_bins, n_frames)
    freqs : np.ndarray
        每个 bin 对应的频率 (Hz)
    times : np.ndarray
        每帧的时间 (秒)
    onset_frames : np.ndarray
        起始帧索引
    onset_times : np.ndarray
        起始时间 (秒)
    sr : int
        采样率
    hop_length : int
        帧移
    n_peaks : int
        每个起始点最多检测的同时音符数
    min_freq, max_freq : float
        频率范围
    threshold_factor : float
        音高检测阈值 (相对于 delta 最大值)

    Returns
    -------
    onset_notes : list of dict
        每个元素对应一个检测到的起始点音符，包含:
        {
            'onset_frame': int,       # 起始帧索引
            'onset_time': float,      # 起始时间（秒）
            'pitch': int,             # MIDI 音符号
            'frequency': float,       # 频率 (Hz)
            'amplitude': float,       # 起始时相对幅度
            'snr': float,             # 信噪比（新能量/旧能量）
        }
    """
    n_frames = cqt.shape[1]

    # 预计算谐波索引表一次（起始点循环内查表）
    harmonic_table = build_harmonic_table(freqs)

    # 计算每个起始点的频谱差
    delta_cqt = _spectral_delta_filter(cqt, onset_frames)

    # 频率掩码
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)

    onset_notes = []

    for idx, onset in enumerate(onset_frames):
        if onset >= n_frames:
            continue

        # 用 delta 频谱做谐波减法检测新进入的音
        delta_frame = delta_cqt[:, onset].copy()
        original_frame = cqt[:, onset].copy()

        # 先试试 delta 有没有能量，如果 delta 太弱就回退到原始频谱
        if np.max(delta_frame) < threshold_factor * np.max(original_frame):
            working_frame = original_frame.copy()
            use_delta = False
        else:
            working_frame = delta_frame.copy()
            use_delta = True

        working_frame[~freq_mask] = 0

        if np.max(working_frame) == 0:
            continue

        # 用谐波模板 + NNLS 检测多音高（查预计算表）
        notes_at_onset = _harmonic_sieve(
            working_frame, freqs,
            n_peaks=n_peaks,
            threshold_factor=threshold_factor,
            harmonic_table=harmonic_table
        )

        for note in notes_at_onset:
            # 计算信噪比：新能量 / 旧能量
            if use_delta and onset > 0:
                base_frame = np.mean(
                    cqt[:, max(0, onset - 3):onset], axis=1
                )
                # 找这个音符对应 bin 的旧能量
                freq_diff = np.abs(freqs / note['frequency'] - 1.0)
                match_idx = np.where(freq_diff < 0.05)[0]
                old_energy = float(np.mean(base_frame[match_idx])) if len(match_idx) > 0 else 0.0
                new_energy = note['amplitude']
                snr = float(new_energy / (old_energy + 1e-10))
            else:
                snr = 1.0

            onset_notes.append({
                'onset_frame': int(onset),
                'onset_time': float(onset_times[idx]),
                'pitch': note['pitch'],
                'frequency': note['frequency'],
                'amplitude': note['amplitude'],
                'snr': snr
            })

    # 按起始时间和幅度排序
    onset_notes.sort(key=lambda n: (n['onset_time'], -n['amplitude']))
    return onset_notes
