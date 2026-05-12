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
    remove_low_amplitude: bool = True,
    melody_split: bool = True
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
    melody_split : bool
        是否进行旋律+伴奏分层识别（默认开启）

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

    # ---- 4. 旋律+伴奏分层识别（核心新功能）----
    if melody_split and len(notes) > 3:
        before = len(notes)
        notes = separate_melody_and_accompaniment(notes)
        # 不打印日志，安静处理

    # ---- 5. 去除谐波成分 ----
    if harmonic_check and len(notes) > 1:
        before = len(notes)
        notes = _remove_harmonics(notes)

    # ---- 6. 去除音高离群值 ----
    before = len(notes)
    notes = _remove_pitch_outliers(notes, outlier_semitones, outlier_window)

    # ---- 7. 限制同时发生的音符数 ----
    if max_simultaneous > 0:
        notes = _limit_simultaneous_notes(notes, max_simultaneous)

    # ---- 8. 限制每拍的音符密度 ----
    if max_notes_per_beat > 0:
        notes = _limit_notes_per_beat(notes, max_notes_per_beat)

    # ---- 9. 最终排序 ----
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


# ============================================================================
#  旋律 + 伴奏分层识别
#  核心策略：先识别主旋律线，再补伴奏音，删除不合理的音
# ============================================================================

MELODY_RANGE = (60, 96)     # C4~C7 主旋律音域
BASS_RANGE = (24, 55)       # C1~G3 低音音域

# 协和音程（以半音为单位，相对于当前旋律音）
CONSONANT_INTERVALS = {0, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 19, 24}
# 不协和音程
DISSONANT_INTERVALS = {1, 2, 6, 10, 11, 13, 14, 18, 20, 21, 22, 23}


def separate_melody_and_accompaniment(notes: list) -> list:
    """
    将检测到的音符分为主旋律 + 伴奏两层。

    策略（模仿人耳听感）：
    1. **先找主旋律**：在每 0.15s 时间窗内，取音域较高、持续较长、
       能量较稳定的音作为旋律候选。旋律线做平滑处理。
    2. **再补低音骨架**：低音区（< C3）中取最稳定的 1~2 个音。
    3. **最后加中音填充**：中音区只保留和声上合理的音
       （与主旋律构成协和音程的三度/五度/八度等），
       丢弃不协和音程（小二度、增四度等）。

    Parameters
    ----------
    notes : list of dict
        音符列表 [{start, end, pitch, velocity}, ...]

    Returns
    -------
    list : 分层优化后的音符列表
    """
    if not notes:
        return []

    notes = [dict(n) for n in notes]
    notes.sort(key=lambda n: (n['start'], -n['pitch']))

    # ---- 1. 按音域分组 ----
    melody_candidates = [n for n in notes
                         if MELODY_RANGE[0] <= n['pitch'] <= MELODY_RANGE[1]]
    bass_candidates = [n for n in notes
                       if BASS_RANGE[0] <= n['pitch'] <= BASS_RANGE[1]]
    mid_candidates = [n for n in notes
                      if BASS_RANGE[1] < n['pitch'] < MELODY_RANGE[0]]

    # ---- 2. 提取主旋律（高音区中每窗选最突出的音） ----
    melody = _extract_melody_line(melody_candidates, window=0.12)

    # ---- 3. 提取低音骨架（低音区中每窗选最稳定的最低音） ----
    bass = _extract_bass_line(bass_candidates, window=0.15)

    # ---- 4. 中音区伴奏（与主旋律和声协调的才保留） ----
    mid_notes = _filter_mid_notes(mid_candidates, melody, window=0.15)

    # ---- 5. 合并且去重 ----
    result = melody + bass + mid_notes
    result.sort(key=lambda n: (n['start'], -n['pitch']))

    # ---- 6. 去重（同一时间完全相同音高的音符） ----
    result = _remove_duplicates(result)

    return result


def _extract_melody_line(
    notes: list,
    window: float = 0.12,
    max_pitch_jump: int = 12,
    min_duration: float = 0.06
) -> list:
    """
    从高音区候选音符中提取主旋律线。

    方法：
    1. 每 window 秒时间窗，取最突出的音（综合评分 = 时长 × 力度 × 音高）
    2. 旋律平滑：相邻窗的旋律音高跳变不能超过 max_pitch_jump
    3. 过滤太短太弱的旋律候选音
    """
    if not notes:
        return []

    t_min = min(n['start'] for n in notes)
    t_max = max(n['end'] for n in notes)
    n_win = max(1, int((t_max - t_min) / window) + 2)

    melody_map = {}  # 时间窗索引 -> {pitch, start, end, score}

    for wi in range(n_win):
        ws = t_min + wi * window
        we = ws + window

        # 该窗内的活动音符
        active = []
        for n in notes:
            if n['start'] < we and n['end'] > ws:
                overlap = min(we, n['end']) - max(ws, n['start'])
                dur = n['end'] - n['start']
                vel = n.get('velocity', 80)
                # 评分：音高越高（旋律常见） + 实际重叠时长 + 力度
                pitch_bonus = 1.0 + (n['pitch'] - 60) / 50.0  # 高音加分
                score = overlap * pitch_bonus * (vel / 127.0)
                active.append((score, n['pitch'], n['start'], n['end'], n))

        if not active:
            continue

        # 选评分最高的
        active.sort(reverse=True)
        best = active[0]
        melody_map[wi] = {
            'pitch': best[1],
            'start': best[2],
            'end': best[3],
            'score': best[0]
        }

    # ---- 旋律平滑：移除跳变过大的离群旋律点 ----
    sorted_indices = sorted(melody_map.keys())

    # 第一次遍历：标记离群点
    outlier_indices = set()
    for i in range(1, len(sorted_indices) - 1):
        prev_idx = sorted_indices[i - 1]
        curr_idx = sorted_indices[i]
        next_idx = sorted_indices[i + 1]

        curr_pitch = melody_map[curr_idx]['pitch']
        prev_pitch = melody_map[prev_idx]['pitch']
        next_pitch = melody_map[next_idx]['pitch']

        # 如果当前音高与前后都相差很大，且自己持续时间短 → 离群
        jump_to_prev = abs(curr_pitch - prev_pitch)
        jump_to_next = abs(curr_pitch - next_pitch)
        curr_dur = melody_map[curr_idx]['end'] - melody_map[curr_idx]['start']

        if jump_to_prev > max_pitch_jump and jump_to_next > max_pitch_jump \
           and curr_dur < 0.15:
            outlier_indices.add(curr_idx)

    # 标记孤立点（前后都没有旋律点，自身音高异常）
    for i in range(len(sorted_indices)):
        curr_idx = sorted_indices[i]
        # 检查周围有没有其他旋律点支撑
        curr_time = t_min + curr_idx * window
        nearby = [idx for idx in sorted_indices
                  if idx != curr_idx
                  and abs((t_min + idx * window) - curr_time) < 0.3]
        if len(nearby) == 0:
            curr_pitch = melody_map[curr_idx]['pitch']
            curr_dur = melody_map[curr_idx]['end'] - melody_map[curr_idx]['start']
            if curr_dur < 0.1:
                outlier_indices.add(curr_idx)

    # ---- 将旋律点转为音符列表 ----
    result = []
    prev_pitch = None
    merge_current = None

    for idx in sorted_indices:
        if idx in outlier_indices:
            continue

        entry = melody_map[idx]
        pitch = entry['pitch']
        start = entry['start']
        end = entry['end']

        # 过滤超短旋律音
        if end - start < min_duration:
            continue

        # 与上一个旋律音合并（同音高或很近且时间相邻）
        if prev_pitch is not None and merge_current is not None:
            if pitch == prev_pitch and start - merge_current['end'] < 0.15:
                merge_current['end'] = max(merge_current['end'], end)
                continue

        note = {
            'start': start,
            'end': end,
            'pitch': pitch,
            'velocity': 85  # 旋律音力度稍高
        }
        result.append(note)
        merge_current = note
        prev_pitch = pitch

    return result


def _extract_bass_line(
    notes: list,
    window: float = 0.15,
    min_duration: float = 0.08
) -> list:
    """
    从低音区候选音符中提取低音骨架。

    每时间窗取最低、最稳定的 1~2 个音。
    低音追求简洁稳定，不做过多装饰。
    """
    if not notes:
        return []

    # 只取音高最低的部分
    notes = sorted(notes, key=lambda n: n['pitch'])
    # 只保留最低的 3 个不同音高
    unique_pitches = []
    for n in notes:
        if not any(abs(n['pitch'] - p) <= 2 for p in unique_pitches):
            unique_pitches.append(n['pitch'])
        if len(unique_pitches) >= 3:
            break
    notes = [n for n in notes if any(abs(n['pitch'] - p) <= 2 for p in unique_pitches)]

    t_min = min(n['start'] for n in notes)
    t_max = max(n['end'] for n in notes)
    n_win = max(1, int((t_max - t_min) / window) + 2)

    bass_map = {}  # 时间窗索引 -> [(pitch, start, end)]

    for wi in range(n_win):
        ws = t_min + wi * window
        we = ws + window

        active = []
        for n in notes:
            if n['start'] < we and n['end'] > ws:
                overlap = min(we, n['end']) - max(ws, n['start'])
                dur = n['end'] - n['start']
                # 评分：越低音越好，越长越好
                bass_bonus = 1.0 + (55 - n['pitch']) / 30.0  # 低音加分
                score = overlap * dur * bass_bonus
                active.append((score, n['pitch'], n['start'], n['end']))

        if not active:
            continue

        # 按评分排序，取最强的 1~2 个
        active.sort(reverse=True)
        top_n = min(2, len(active))
        picks = []
        for j in range(top_n):
            picks.append({
                'pitch': active[j][1],
                'start': active[j][2],
                'end': active[j][3]
            })

        # 去重：如果两个音音高太近只保留更强的
        if len(picks) > 1 and abs(picks[0]['pitch'] - picks[1]['pitch']) <= 3:
            picks = [picks[0]]

        bass_map[wi] = picks

    # 转为音符列表，合并相邻同音高
    result = []
    sorted_indices = sorted(bass_map.keys())
    prev_notes = []

    for idx in sorted_indices:
        picks = bass_map[idx]
        for pick in picks:
            pitch = pick['pitch']
            start = pick['start']
            end = pick['end']

            if end - start < min_duration:
                continue

            # 合并
            merged = False
            for r in reversed(result):
                if r['pitch'] == pitch and start - r['end'] < 0.15:
                    r['end'] = max(r['end'], end)
                    merged = True
                    break

            if not merged:
                result.append({
                    'start': start,
                    'end': end,
                    'pitch': pitch,
                    'velocity': 70
                })

    return result


def _filter_mid_notes(
    mid_notes: list,
    melody_notes: list,
    window: float = 0.15
) -> list:
    """
    过滤中音区伴奏：只保留与主旋律构成协和音程的音。

    规则：
    - 如果某时间窗内没有主旋律，中音区音全部保留（但有密度限制）
    - 如果有主旋律，检查每个中音与旋律的音程关系
    - 协和音程（三度/五度/八度等）保留
    - 不协和音程（小二度/增四度等）视力度和时长决定
    """
    if not mid_notes:
        return []

    # 构建旋律音的时间索引
    melody_by_time = []
    for m in melody_notes:
        melody_by_time.append((m['start'], m['end'], m['pitch']))

    t_min = min(m['start'] for m in mid_notes)
    t_max = max(m['end'] for m in mid_notes)
    n_win = max(1, int((t_max - t_min) / window) + 2)

    kept_ids = set()

    for wi in range(n_win):
        ws = t_min + wi * window
        we = ws + window

        # 该窗内的中音
        mid_active = [(n, n['end'] - n['start'], n.get('velocity', 80))
                      for n in mid_notes
                      if n['start'] < we and n['end'] > ws]

        if not mid_active:
            continue

        # 该窗内是否有旋律音
        melody_here = [m[2] for m in melody_by_time
                      if not (m[0] >= we or m[1] <= ws)]

        if not melody_here:
            # 没有旋律：密度限制保留
            mid_active.sort(key=lambda x: x[1], reverse=True)
            kept_count = min(3, len(mid_active))
            for j in range(kept_count):
                kept_ids.add(id(mid_active[j][0]))
            continue

        # 有旋律：检查每个中音与旋律的和声关系
        melody_pitch = max(melody_here)  # 取最高旋律音为参考

        for n, dur, vel in mid_active:
            # 计算与旋律的音程（半音数，取绝对值，再取模 12 得到音程类别）
            interval = abs(n['pitch'] - melody_pitch) % 12

            # 协和音程直接保留
            if interval in {0, 3, 4, 5, 7, 8, 9}:
                kept_ids.add(id(n))
                continue

            # 比较不协和但很强/很长的音：保留（可能是故意的色彩音）
            if interval in {2, 10}:  # 大二度/小七度
                score = dur * (vel / 127.0)
                if score > 0.1:
                    kept_ids.add(id(n))
                continue

            # 极不协和音程（小二度/增四度）：只有足够强才保留
            if interval in {1, 6, 11}:
                score = dur * (vel / 127.0)
                if score > 0.15:
                    kept_ids.add(id(n))
                continue

    result = [n for n in mid_notes if id(n) in kept_ids]
    return result


def _remove_duplicates(notes: list) -> list:
    """移除完全相同（时间+音高）的重复音符"""
    seen = set()
    result = []
    for n in sorted(notes, key=lambda x: (x['start'], x['pitch'])):
        key = (round(n['start'], 2), round(n['end'], 2), n['pitch'])
        if key not in seen:
            seen.add(key)
            result.append(n)
    return result
