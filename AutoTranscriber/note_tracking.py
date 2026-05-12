"""音符追踪模块 — 将逐帧音高估计结果连接成音符事件"""

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
    将逐帧音高检测结果追踪为连续的音符事件。

    算法：
    1. 在检测到的起始点之间，观察每帧的音高
    2. 如果相邻帧的相同音高持续出现，合并为一个音符
    3. 忽略持续时间过短的音符
    4. 输出 (起始时间, 结束时间, 音高, 力度)

    Parameters
    ----------
    frame_notes : list of list of dict
        每帧的音高检测结果
    onset_frames : np.ndarray
        起始帧索引
    onset_times : np.ndarray
        起始时间（秒）
    times : np.ndarray
        所有帧的时间点（秒）
    sr : int
        采样率
    hop_length : int
        帧移
    min_note_duration : int
        最小音符持续帧数（低于此值视为噪声）
    pitch_hysteresis : int
        音高追踪时的允许波动范围（半音）
    velocity_scale : float
        力度缩放系数

    Returns
    -------
    notes : list of dict
        音符事件列表，每个包含:
        {
            'start': float,      # 起始时间（秒）
            'end': float,        # 结束时间（秒）
            'pitch': int,        # MIDI 音符号
            'velocity': int      # 力度 (0-127)
        }
    """
    n_frames = len(frame_notes)
    if n_frames == 0:
        return []

    # 不使用 onset 分段，而是在整个时间轴上独立追踪每个音高
    # 这种方法对和弦更友好（和弦音同时起落）
    all_notes = _track_continuous(
        frame_notes, 0, n_frames - 1,
        times, min_note_duration, velocity_scale,
        min_amplitude=min_amplitude
    )

    # 合并重叠或相邻的相同音高音符
    all_notes = _merge_notes(all_notes)

    # 过滤过短音符（按实际时间）
    min_sec = max(0.05, min_note_duration * (hop_length / sr))
    all_notes = [n for n in all_notes if (n['end'] - n['start']) >= min_sec]

    # 按起始时间排序
    all_notes.sort(key=lambda n: n['start'])

    return all_notes


def _track_continuous(frame_notes, start_frame, end_frame,
                      times, min_duration, vel_scale, min_amplitude=0.08):
    """
    在连续帧区间内追踪音符。
    对每个检测到的音高，检查其是否连续出现若干帧。
    """
    # 构建每个音高的活跃帧序列
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

    # 将活跃帧序列分割为连续的块
    notes = []
    for pitch, active_list in pitch_active_frames.items():
        # 按帧号排序
        active_list.sort(key=lambda x: x['frame'])

        # 分割连续块（帧号不连续则断开）
        chunks = []
        current_chunk = [active_list[0]]

        for i in range(1, len(active_list)):
            if active_list[i]['frame'] == active_list[i - 1]['frame'] + 1:
                current_chunk.append(active_list[i])
            else:
                chunks.append(current_chunk)
                current_chunk = [active_list[i]]
        chunks.append(current_chunk)

        # 保留足够长的块
        for chunk in chunks:
            if len(chunk) >= min_duration:
                start = times[chunk[0]['frame']]
                end = times[chunk[-1]['frame']]
                # 计算平均力度
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
    """
    合并时间上相邻或相近的相同音高音符。
    gap_threshold: 两个音符间隔小于此值则合并（秒）
    """
    if not notes:
        return []

    # 按音高分组
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
