"""
感知滤波模块 — 追求听感相似性，放弃绝对还原。

核心策略：
1. 去除明显偏离主旋律的离群音（音高突变 > 12 半音且孤立）
2. 去除谐波成分（某个音是另一个更强音的整数倍 → 只保留基频）
3. 合并相邻的相同/相近音高，用时长代替碎音
4. 过滤超短音符（< 0.05s）
5. 过滤过低振幅的音符
"""

import numpy as np
from collections import defaultdict


def perceptual_filter(
    notes: list,
    outlier_semitones: int = 12,
    outlier_window: float = 0.5,
    min_duration: float = 0.05,
    min_velocity: int = 1,
    merge_gap: float = 0.08,
    merge_semitones: int = 2,
    harmonic_check: bool = True,
    max_simultaneous: int = 6,
    max_notes_per_beat: int = 8,
    remove_low_amplitude: bool = True
) -> list:
    """
    对检测到的音符列表进行感知滤波。

    Parameters
    ----------
    notes : list of dict
        原始音符列表 [{start, end, pitch, velocity}, ...]
    outlier_semitones : int
        偏离主旋律多少半音视为离群音 (默认12 = 八度)
    outlier_window : float
        判断离群的局部时间窗 (秒)
    min_duration : float
        最小音符时长（秒），短于此的删除
    min_velocity : int
        最小力度，低于此的删除
    merge_gap : float
        相同音高合并的最大间隔（秒）
    merge_semitones : int
        合并时允许的最大音高差（半音）
    harmonic_check : bool
        是否检查谐波并移除
    max_simultaneous : int
        同一时刻允许的最大同时音符数
    max_notes_per_beat : int
        每拍的音符数上限
    remove_low_amplitude : bool
        是否自动剔除低评分音符

    Returns
    -------
    filtered_notes : list of dict
        滤波后的音符列表
    """
    if not notes:
        return []

    notes = [dict(n) for n in notes]

    # ---- 1. 过滤超短音符 ----
    notes = [n for n in notes if n['end'] - n['start'] >= min_duration]

    # ---- 2. 过滤过低力度 ----
    notes = [n for n in notes if n.get('velocity', 80) >= min_velocity]

    # ---- 3. 合并相邻相同/相近音高 ----
    notes = _merge_nearby_notes(notes, merge_gap, merge_semitones)

    # ---- 4. 去除谐波成分 ----
    if harmonic_check and len(notes) > 1:
        before = len(notes)
        notes = _remove_harmonics(notes)
        if len(notes) < before:
            pass  # 安静地移除

    # ---- 5. 去除音高离群值 ----
    before = len(notes)
    notes = _remove_pitch_outliers(notes, outlier_semitones, outlier_window)
    if len(notes) < before:
        pass  # 安静地移除

    # ---- 6. 限制同时发生的音符数 ----
    if max_simultaneous > 0:
        notes = _limit_simultaneous_notes(notes, max_simultaneous)

    # ---- 7. 限制每拍的音符密度 ----
    if max_notes_per_beat > 0:
        notes = _limit_notes_per_beat(notes, max_notes_per_beat)

    # ---- 8. 最终排序 ----
    notes.sort(key=lambda n: (n['start'], -n['pitch']))

    return notes


def _merge_nearby_notes(
    notes: list,
    gap_threshold: float = 0.08,
    pitch_tolerance: int = 2
) -> list:
    """
    合并时间上相邻、音高相近的音符。
    
    规则：
    - 相同音高、间隔 < gap_threshold → 合并
    - 音高差 ≤ pitch_tolerance，间隔很短 → 取较长的那条
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))
    merged = []
    current = notes[0].copy()

    for n in notes[1:]:
        gap = n['start'] - current['end']
        pdiff = abs(n['pitch'] - current['pitch'])

        # 同音高合并
        if pdiff == 0 and gap <= gap_threshold:
            current['end'] = max(current['end'], n['end'])
            current['velocity'] = max(current['velocity'], n.get('velocity', 80))
            continue

        # 相近音高、间隔很短 → 取更长的
        if pdiff <= pitch_tolerance and gap <= gap_threshold * 0.5:
            curr_dur = current['end'] - current['start']
            this_dur = n['end'] - n['start']
            if this_dur > curr_dur:
                current['pitch'] = n['pitch']
                current['end'] = max(current['end'], n['end'])
                current['velocity'] = max(current['velocity'], n.get('velocity', 80))
            continue

        merged.append(current)
        current = n.copy()

    merged.append(current)
    return merged


def _remove_harmonics(notes: list, max_harmonic: int = 6) -> list:
    """
    去除谐波成分。

    如果一个音是另一个更强音的 2x ~ 6x 倍频，且当前音振幅更低，
    则这个音很可能是基频的泛音，应该被移除。

    保留规则：
    - 基频的幅度必须显著高于谐波（基频更强）
    - 八度关系特殊对待（八度有时是独立音）
    """
    if not notes:
        return []

    # 对重叠的音符分组检查
    notes_sorted = sorted(notes, key=lambda n: (n['start'], n['end']))
    result = list(notes_sorted)
    to_remove = set()

    for i, n1 in enumerate(notes_sorted):
        if id(n1) in to_remove:
            continue
        for j, n2 in enumerate(notes_sorted):
            if i == j or id(n2) in to_remove:
                continue

            # 检查时间是否重叠
            if not (n1['start'] < n2['end'] and n2['start'] < n1['end']):
                continue

            # 检查频率关系（基频 vs 谐波）
            higher = n1 if n1['pitch'] > n2['pitch'] else n2
            lower = n2 if n1['pitch'] > n2['pitch'] else n1

            semitone_diff = higher['pitch'] - lower['pitch']
            # 频率比 = 2^(半音差/12)
            # 检查是否是 2~6 倍频关系 (约 +12,+19,+24,+28,+31,+34 半音)
            harmonic_intervals = {
                12: '八度',    # 2x
                19: '五度',    # 3x
                24: '双八度',  # 4x
                28: '三度',    # 5x
                31: '五度',    # 6x - approximate
            }

            is_harmonic = False
            for interval, _ in harmonic_intervals.items():
                if abs(semitone_diff - interval) <= 2:
                    is_harmonic = True
                    break

            if not is_harmonic:
                continue

            # 如果基频的强度不低于谐波，移除谐波
            lower_vel = lower.get('velocity', 80)
            higher_vel = higher.get('velocity', 80)

            # 对八度保留宽松政策（八度音经常是故意弹的）
            if semitone_diff == 12 or abs(semitone_diff - 12) <= 1:
                if higher_vel < lower_vel * 0.4:  # 只有谐波很弱才移除
                    to_remove.add(id(higher))
            else:
                if higher_vel <= lower_vel:  # 非八度谐波，只要基频更强就移除
                    to_remove.add(id(higher))

    result = [n for n in result if id(n) not in to_remove]
    return result


def _remove_pitch_outliers(
    notes: list,
    max_jump: int = 12,
    window: float = 0.5
) -> list:
    """
    去除音高离群值。

    在局部时间窗内，如果某个音的音高与周围音符的中位数差距超过 max_jump 半音，
    且这个音是孤立的（周围没有其他同音高的音符支撑），则视为离群值删除。

    特殊处理：
    - 短音符更容易被判定为离群值
    - 持续较长的音即使音高高，也可能是有意义的（如副歌高潮）
    """
    if not notes:
        return []

    notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))

    to_remove = set()

    for i, n in enumerate(notes):
        dur = n['end'] - n['start']

        # 时长较长的音更可能是有效音，放宽判断
        dur_factor = 1.0 if dur < 0.15 else (0.5 if dur < 0.3 else 0.0)
        if dur_factor == 0.0:
            continue  # 长音不删除

        adjusted_jump = max_jump * (1.0 + dur_factor)

        # 在时间窗内找邻居的音高中位数
        t = n['start']
        neighbors = []
        for j, neighbor in enumerate(notes):
            if j == i:
                continue
            if abs(neighbor['start'] - t) <= window:
                neighbors.append(neighbor['pitch'])

        if not neighbors:
            continue

        median_pitch = float(np.median(neighbors))
        deviation = abs(n['pitch'] - median_pitch)

        if deviation > adjusted_jump:
            to_remove.add(id(n))

    result = [n for n in notes if id(n) not in to_remove]
    return result


def _limit_simultaneous_notes(notes: list, max_simultaneous: int = 6) -> list:
    """
    限制同一时刻最多响起的音符数量。
    如果某时刻超过 max_simultaneous 个音同时响，只保留最强的几个。
    """
    if not notes:
        return []

    # 找出所有时间点并按紧密度评估
    t_min = min(n['start'] for n in notes)
    t_max = max(n['end'] for n in notes)
    window = 0.1  # 100ms 采样窗
    n_win = int((t_max - t_min) / window) + 1

    # 统计每个音被裁剪的次数
    removal_score = defaultdict(float)

    for i in range(n_win):
        ws = t_min + i * window
        we = ws + window

        active = [(n, n['end'] - n['start'], n.get('velocity', 80))
                  for n in notes
                  if n['start'] < we and n['end'] > ws]

        if len(active) <= max_simultaneous:
            continue

        # 太多音同时响了，标记那些「最短力最小的」
        # 评分：优先保留音域两端的音（旋律+低音骨架）
        scored = []
        for n, dur, vel in active:
            # 高音和低音更有保留价值
            if n['pitch'] >= 67:  # 高音区 → 旋律
                bonus = 2.0
            elif n['pitch'] < 48:  # 低音区 → 低音骨架
                bonus = 1.5
            else:  # 中音区 → 最容易冗余
                bonus = 0.5

            score = dur * bonus * (vel / 127.0)
            scored.append((score, n))

        scored.sort(reverse=True)
        # 保留 top max_simultaneous
        kept = set(id(s[1]) for s in scored[:max_simultaneous])
        for score, n in scored[max_simultaneous:]:
            removal_score[id(n)] += 1.0

    # 移除被标记删除次数 >= 2 的音符
    result = [n for n in notes if removal_score.get(id(n), 0) < 2.0]
    return result


def _limit_notes_per_beat(notes: list, max_per_beat: int = 8) -> list:
    """
    限制每拍的音符密度。
    如果某 0.5s 窗口内的音符数过多，删除评分最低的音符。

    这有助于防止录音环境中的杂音被误识别为音符。
    """
    if not notes:
        return []

    beat_sec = 0.5  # 120 BPM 每拍 0.5s
    t_min = min(n['start'] for n in notes)
    t_max = max(n['end'] for n in notes)
    n_beats = int((t_max - t_min) / beat_sec) + 1

    removal_score = defaultdict(float)

    for bi in range(n_beats):
        bs = t_min + bi * beat_sec
        be = bs + beat_sec

        active = [n for n in notes
                  if n['start'] < be and n['end'] > bs]

        if len(active) <= max_per_beat:
            continue

        # 评分：时长 × 力度
        scored = []
        for n in active:
            dur = n['end'] - n['start']
            vel = n.get('velocity', 80)
            # 力度权重
            score = dur * (vel / 127.0)
            scored.append((score, n))

        scored.sort(reverse=True)
        for score, n in scored[max_per_beat:]:
            removal_score[id(n)] += 1.0

    result = [n for n in notes if removal_score.get(id(n), 0) < 1.0]
    return result
