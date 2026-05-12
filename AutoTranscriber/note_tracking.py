"""音符追踪模块 — 基于起始点的音符时长追踪"""

import numpy as np
from collections import defaultdict


def track_notes(frame_notes: list, onset_frames: np.ndarray,
                onset_times: np.ndarray, times: np.ndarray,
                sr: int, hop_length: int = 512,
                min_note_duration: int = 4,
                pitch_hysteresis: int = 1,
                velocity_scale: float = 80.0,
                min_amplitude: float = 0.1) -> list:
    """
    [旧方法] 将逐帧音高检测结果追踪为连续的音符事件。
    保留用于兼容，新方法请用 track_notes_onset_driven。
    """
    return _track_notes_legacy(
        frame_notes, onset_frames, onset_times, times,
        sr, hop_length, min_note_duration,
        pitch_hysteresis, velocity_scale, min_amplitude
    )


def _track_notes_legacy(frame_notes, onset_frames, onset_times, times,
                        sr, hop_length, min_note_duration,
                        pitch_hysteresis, velocity_scale, min_amplitude):
    """原 track_notes 实现"""
    n_frames = len(frame_notes)
    if n_frames == 0:
        return []

    all_notes = _track_continuous(
        frame_notes, 0, n_frames - 1,
        times, min_note_duration, velocity_scale,
        min_amplitude=min_amplitude
    )

    all_notes = _merge_notes(all_notes)

    min_sec = max(0.05, min_note_duration * (hop_length / sr))
    all_notes = [n for n in all_notes if (n['end'] - n['start']) >= min_sec]

    all_notes.sort(key=lambda n: n['start'])
    return all_notes


# ============================================================================
#  新方法：基于起始点的音符追踪
#  核心思路：
#  1. 在起始点检测到音高后，向前追踪该音的振幅变化
#  2. 当振幅衰减到起始振幅的一定比例以下时，认为该音结束
#  3. 这比逐帧检测更符合人耳感知（起音→衰减→延音）
# ============================================================================

def track_notes_onset_driven(
    onset_notes: list,
    cqt: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    sr: int,
    hop_length: int = 512,
    max_gap_frames: int = 3,
    decay_ratio: float = 0.25,
    min_note_frames: int = 2,
    max_note_seconds: float = 4.0,
    velocity_scale: float = 80.0,
    snr_threshold: float = 0.5
) -> list:
    """
    基于起始点的音符追踪。

    对新方法 `estimate_pitches_onset_driven` 的输出进行处理，
    追踪每个起始音在时间上的持续情况。

    算法：
    1. 对每个在起始点检测到的音，找到它在 CQT 谱中的对应频带
    2. 从起始帧开始向前扫描，监测该频带的振幅变化
    3. 当振幅衰减到 < decay_ratio × 起始振幅 时，标记该音结束
    4. 如果中间出现空隙（gap），允许短时间恢复
    5. 在另一个起始点检测到相同音高时，优先闭合旧音再开新音

    Parameters
    ----------
    onset_notes : list of dict
        estimate_pitches_onset_driven 的输出
    cqt : np.ndarray
        CQT 频谱 (n_bins, n_frames)
    freqs : np.ndarray
        每个 bin 对应的频率 (Hz)
    times : np.ndarray
        每帧的时间 (秒)
    sr : int
        采样率
    hop_length : int
        帧移
    max_gap_frames : int
        振幅低于阈值后，允许多少帧恢复（防颤音误断）
    decay_ratio : float
        振幅衰减到起始值的多少比例时判定结束（0.25 = 衰减到25%时结束）
    min_note_frames : int
        最小音符持续帧数（过滤短噪声）
    max_note_seconds : float
        最大持续时长（防止无限延长，4秒适合大多数乐器）
    velocity_scale : float
        力度缩放系数
    snr_threshold : float
        信噪比阈值，低于此值的起始音视为噪声丢弃

    Returns
    -------
    notes : list of dict
        [{'start': float, 'end': float, 'pitch': int, 'velocity': int}, ...]
    """
    if not onset_notes or cqt.size == 0:
        return []

    n_frames = cqt.shape[1]

    # 按起始时间排序
    onset_notes = sorted(onset_notes, key=lambda n: n['onset_time'])

    # ---- 剔除低信噪比的起始音（可能是噪声） ----
    onset_notes = [n for n in onset_notes if n['snr'] >= snr_threshold]

    # ---- 追踪每个起始音的持续时间 ----
    notes = []
    active_pitches = {}  # pitch -> {start_frame, start_amp}

    for on in onset_notes:
        onset_frame = on['onset_frame']
        pitch = on['pitch']
        onset_amp = on['amplitude']

        # 如果这个音高还在响，先关掉旧的
        if pitch in active_pitches:
            prev = active_pitches.pop(pitch)
            end_time = times[onset_frame]  # 新音的开始即为旧音的结束
            if (end_time - times[prev['start_frame']]) >= min_note_frames * (hop_length / sr):
                velocity = min(127, max(1, int(prev['start_amp'] * velocity_scale)))
                notes.append({
                    'start': float(times[prev['start_frame']]),
                    'end': float(end_time),
                    'pitch': int(pitch),
                    'velocity': velocity
                })

        # ---- 向前追踪当前音的振幅衰减 ----
        # 找到该音高在 CQT 中的频带范围
        freq = midi_to_hz(pitch)
        # 半音带宽 (±1 半音)
        freq_low = midi_to_hz(pitch - 1)
        freq_high = midi_to_hz(pitch + 1)
        band_mask = (freqs >= freq_low) & (freqs <= freq_high)
        band_indices = np.where(band_mask)[0]

        if len(band_indices) == 0:
            continue

        # 起始振幅 = 该频带在起始帧的最大能量
        start_amp_frame = float(np.max(cqt[band_indices, onset_frame]))
        if start_amp_frame == 0:
            continue

        # 从起始帧开始向前追踪
        end_frame = onset_frame
        below_threshold_count = 0

        max_end_frame = min(n_frames - 1,
                            onset_frame + int(max_note_seconds * sr / hop_length))

        for t in range(onset_frame + 1, max_end_frame + 1):
            # 该频带在当前帧的最高能量
            current_amp = float(np.max(cqt[band_indices, t]))

            # 如果振幅 > decay_ratio × 起始振幅，认为还在响
            if current_amp > decay_ratio * start_amp_frame:
                end_frame = t
                below_threshold_count = 0
            else:
                below_threshold_count += 1
                # 如果连续多帧低于阈值，且不是短暂间隙，结束
                if below_threshold_count > max_gap_frames:
                    break

        # 记录该音为活跃状态
        if end_frame - onset_frame >= min_note_frames:
            active_pitches[pitch] = {
                'start_frame': onset_frame,
                'start_amp': onset_amp,
                'end_frame': end_frame
            }

    # ---- 处理仍在响的剩余音符 ----
    for pitch, info in active_pitches.items():
        end_time = float(times[info['end_frame']])
        start_time = float(times[info['start_frame']])
        if (end_time - start_time) >= min_note_frames * (hop_length / sr):
            velocity = min(127, max(1, int(info['start_amp'] * velocity_scale)))
            notes.append({
                'start': start_time,
                'end': end_time,
                'pitch': int(pitch),
                'velocity': velocity
            })

    # ---- 合并重叠或相近的相同音高音符 ----
    notes = _merge_onset_notes(notes)

    # 按起始时间排序
    notes.sort(key=lambda n: n['start'])
    return notes


def _merge_onset_notes(notes: list, gap_threshold: float = 0.05) -> list:
    """
    合并时间上相邻的相同音高音符。
    与旧 _merge_notes 功能类似，但专门处理起始点追踪产生的音符结构。
    """
    if not notes:
        return []

    by_pitch = defaultdict(list)
    for n in notes:
        by_pitch[n['pitch']].append(n)

    merged = []
    for pitch, note_list in by_pitch.items():
        note_list.sort(key=lambda n: n['start'])
        current = note_list[0].copy()

        for n in note_list[1:]:
            # 如果间隔小于阈值，合并
            if n['start'] - current['end'] <= gap_threshold:
                current['end'] = max(current['end'], n['end'])
                current['velocity'] = max(current['velocity'], n['velocity'])
            else:
                merged.append(current)
                current = n.copy()
        merged.append(current)

    return merged


def midi_to_hz(midi_note: int) -> float:
    """MIDI 音符号转频率 (Hz)"""
    return 440.0 * (2 ** ((midi_note - 69) / 12.0))


# ---------- 以下为旧方法的辅助函数（保留兼容） ----------

def _track_continuous(frame_notes, start_frame, end_frame,
                      times, min_duration, vel_scale, min_amplitude=0.08):
    """[旧] 在连续帧区间内追踪音符"""
    pitch_active_frames = defaultdict(list)
    for t in range(start_frame, min(end_frame, len(frame_notes))):
        notes_at_t = frame_notes[t]
        for note in notes_at_t:
            pitch = note['pitch']
            amp = note.get('amplitude', 0.0)
            if amp < min_amplitude:
                continue
            pitch_active_frames[pitch].append({
                'frame': t,
                'amplitude': amp
            })

    notes = []
    for pitch, active_list in pitch_active_frames.items():
        active_list.sort(key=lambda x: x['frame'])
        chunks = []
        current_chunk = [active_list[0]]
        for i in range(1, len(active_list)):
            if active_list[i]['frame'] == active_list[i - 1]['frame'] + 1:
                current_chunk.append(active_list[i])
            else:
                chunks.append(current_chunk)
                current_chunk = [active_list[i]]
        chunks.append(current_chunk)

        for chunk in chunks:
            if len(chunk) >= min_duration:
                start = times[chunk[0]['frame']]
                end = times[chunk[-1]['frame']]
                avg_amp = np.mean([c['amplitude'] for c in chunk])
                velocity = min(127, max(1, int(avg_amp * vel_scale)))
                notes.append({
                    'start': float(start),
                    'end': float(end),
                    'pitch': int(pitch),
                    'velocity': velocity
                })
    return notes


def _merge_notes(notes, gap_threshold: float = 0.05):
    """[旧] 合并相邻相同音高音符"""
    if not notes:
        return []
    by_pitch = defaultdict(list)
    for n in notes:
        by_pitch[n['pitch']].append(n)
    merged = []
    for pitch, note_list in by_pitch.items():
        note_list.sort(key=lambda n: n['start'])
        current = note_list[0].copy()
        for n in note_list[1:]:
            if n['start'] - current['end'] <= gap_threshold:
                current['end'] = max(current['end'], n['end'])
                current['velocity'] = max(current['velocity'], n['velocity'])
            else:
                merged.append(current)
                current = n.copy()
        merged.append(current)
    return merged
