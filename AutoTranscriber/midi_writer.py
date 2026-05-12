"""MIDI 文件生成与合并模块"""

import pretty_midi
import numpy as np
from collections import defaultdict


def write_midi(notes: list, output_path: str,
               tempo: float = 120.0,
               program: int = 0) -> str:
    """写入单轨 MIDI 文件"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instr = pretty_midi.Instrument(program=program)
    for n in notes:
        if n['start'] < 0 or n['end'] <= n['start'] or not (0 < n['pitch'] < 128):
            continue
        mn = pretty_midi.Note(
            velocity=n['velocity'], pitch=n['pitch'],
            start=n['start'], end=n['end']
        )
        instr.notes.append(mn)
    midi.instruments.append(instr)
    midi.write(output_path)
    return output_path


def merge_tracks_to_piano(track_notes_list: list,
                          max_simultaneous: int = 4) -> list:
    """
    将多轨合并为干净可弹的钢琴谱。

    核心策略："旋律+低音骨干" 
    - 人声轨 = 旋律 → 取每时刻最强音高，做平滑
    - 伴奏轨 = 低音骨干 → 只保留最低的 1~2 个音
    - 保留每一个和弦的根音和次低音作为低音进行
    - 不要和弦填充，不要中音区杂音
    """
    # ---- 1. 合并所有音符，给每帧选一个最高音和一个最低音 ----
    all_raw = []
    for notes, ttype in track_notes_list:
        smoothed = _smooth_pitches(list(notes)) if ttype == 'vocal' else list(notes)
        for n in smoothed:
            all_raw.append(n)

    if not all_raw:
        return []

    # 去重
    seen = set()
    unique = []
    for n in sorted(all_raw, key=lambda x: (x['start'], -x['pitch'])):
        key = (round(n['start'], 2), round(n['end'], 2), n['pitch'])
        if key not in seen:
            seen.add(key)
            unique.append(n)

    # ---- 2. 分时间窗，每窗取最高音（旋律）和最低音（低音骨干） ----
    t_min = min(n['start'] for n in unique)
    t_max = max(n['end'] for n in unique)
    window = 0.15  # 150ms 窗
    n_win = int((t_max - t_min) / window) + 1

    melody_map = {}  # time_key -> (pitch, start, end)
    bass_map = {}    # time_key -> [(pitch, start, end)]

    for i in range(n_win):
        ws = t_min + i * window
        we = ws + window

        active = [n for n in unique if n['start'] < we and n['end'] > ws]
        if not active:
            continue

        active.sort(key=lambda x: x['pitch'])
        lowest = active[0]
        highest = active[-1]

        tk = round(ws, 1)

        # 旋律 = 最高音
        if tk not in melody_map or \
           (highest['end'] - highest['start']) > \
           (melody_map[tk][2] - melody_map[tk][1]):
            melody_map[tk] = (highest['pitch'], highest['start'], highest['end'])

        # 低音 = 最低音
        if tk not in bass_map:
            bass_map[tk] = []
        # 检查是否已存在相同音高
        if not any(abs(lowest['pitch'] - p) < 3 for p, _, _ in bass_map[tk]):
            bass_map[tk].append((lowest['pitch'], lowest['start'], lowest['end']))

        # 如果最高音和最低音不同，可能加第二个低音
        if lowest['pitch'] != highest['pitch']:
            if len(active) >= 3:
                second_lowest = active[1]
                if not any(abs(second_lowest['pitch'] - p) < 3 for p, _, _ in bass_map[tk]):
                    bass_map[tk].append((second_lowest['pitch'], second_lowest['start'], second_lowest['end']))

    # ---- 3. 转为音符列表 ----
    result = []
    sorted_keys = sorted(set(list(melody_map.keys()) + list(bass_map.keys())))

    for key in sorted_keys:
        if key in melody_map:
            pitch, start, end = melody_map[key]
            if end - start >= 0.08:
                # 与上一个同音高的旋律合并
                if result and result[-1]['pitch'] == pitch and \
                   abs(result[-1]['start'] - start) < 0.2:
                    result[-1]['end'] = max(result[-1]['end'], end)
                else:
                    result.append({'start': start, 'end': end, 'pitch': pitch, 'velocity': 80})

        if key in bass_map:
            for pitch, start, end in bass_map[key]:
                if end - start >= 0.08 and \
                   not any(abs(pitch - r['pitch']) <= 1 and 
                           abs(r['start'] - start) < 0.1 for r in result):
                    # 与上一个同音高低音合并
                    if result and result[-1]['pitch'] == pitch and \
                       abs(result[-1]['start'] - start) < 0.2:
                        result[-1]['end'] = max(result[-1]['end'], end)
                    else:
                        result.append({'start': start, 'end': end, 'pitch': pitch, 'velocity': 70})

    result.sort(key=lambda n: (n['start'], -n['pitch']))

    melody_count = sum(1 for n in result if n['pitch'] >= 55)
    bass_count = sum(1 for n in result if n['pitch'] < 55)
    print(f"      🎹 旋律: {melody_count} + 低音: {bass_count} = {len(result)}")
    return result


def _smooth_pitches(notes: list, min_gap: float = 0.06,
                    max_pitch_jump: int = 2) -> list:
    """
    人声平滑：合并颤音/滑音造成的相邻细碎音符。
    
    原理：
    - 同一音高、间隔 < 0.06s 的音符合并
    - 音高差 ≤ 2、间隔 < 0.06s 的取时长更长的为主音
    - 过滤短于 0.05s 的音符碎片
    """
    if not notes:
        return []
    notes = sorted(notes, key=lambda n: (n['start'], n['pitch']))
    
    merged = []
    cur = dict(notes[0])
    
    for n in notes[1:]:
        gap = n['start'] - cur['end']
        pdiff = abs(n['pitch'] - cur['pitch'])
        
        # 同音高连续或间隔很小 → 合并
        if pdiff == 0 and gap <= min_gap * 3:
            cur['end'] = max(cur['end'], n['end'])
            continue
        
        # 音高接近、间隔很小 → 取更长的那个音
        if pdiff <= max_pitch_jump and gap <= min_gap:
            if n['end'] - n['start'] > cur['end'] - cur['start']:
                cur['pitch'] = n['pitch']
                cur['end'] = n['end']
            continue
        
        merged.append(dict(cur))
        cur = dict(n)
    
    merged.append(dict(cur))
    
    # 过滤短音符
    result = [n for n in merged if n['end'] - n['start'] >= 0.05]
    return result


def _extract_melody(notes: list, min_duration: float = 0.1) -> list:
    """
    从人声轨中提取干净的旋律线。
    
    方法：按 0.1s 时间窗分组，每个时间窗保留
    持续最长的音高作为该时刻的旋律音。
    """
    if not notes:
        return []
    
    notes = sorted(notes, key=lambda n: (n['start'], -n['pitch']))
    
    # 时间窗 0.1s
    t_min = min(n['start'] for n in notes)
    t_max = max(n['end'] for n in notes)
    window = 0.1
    n_win = int((t_max - t_min) / window) + 1
    
    melody_pitches = {}  # round(start,1) -> pitch
    
    for i in range(n_win):
        ws = t_min + i * window
        we = ws + window
        
        # 该窗内最长的音符
        active = [(n['end'] - n['start'], n['pitch'], n)
                  for n in notes if n['start'] < we and n['end'] > ws]
        if not active:
            continue
        
        # 取持续时间最长的
        active.sort(reverse=True)
        _, pitch, best_n = active[0]
        
        time_key = round(ws, 1)
        if time_key not in melody_pitches:
            melody_pitches[time_key] = (pitch, best_n['start'], best_n['end'])
        else:
            # 如果有冲突，取更长的
            existing_dur = melody_pitches[time_key][2] - melody_pitches[time_key][1]
            new_dur = best_n['end'] - best_n['start']
            if new_dur > existing_dur:
                melody_pitches[time_key] = (pitch, best_n['start'], best_n['end'])
    
    # 将旋律音高转为音符
    result = []
    sorted_keys = sorted(melody_pitches.keys())
    
    for i, key in enumerate(sorted_keys):
        pitch, start, end = melody_pitches[key]
        
        # 相邻时间窗相同音高 → 合并
        if i > 0:
            prev_key = sorted_keys[i - 1]
            prev_pitch, prev_start, prev_end = melody_pitches[prev_key]
            if pitch == prev_pitch and key - prev_key <= 0.2:
                result[-1] = {
                    'start': prev_start,
                    'end': max(prev_end, end),
                    'pitch': pitch,
                    'velocity': 80
                }
                continue
        
        # 过滤太短的音符
        if end - start >= min_duration:
            result.append({
                'start': start,
                'end': end,
                'pitch': pitch,
                'velocity': 80
            })
    
    return result


def _extract_bass_line(notes: list, min_duration: float = 0.1) -> list:
    """
    从伴奏轨提取低音骨干。
    
    方法：按时间窗分组，每个时间窗保留最低的 1~2 个音，
    去掉高音区的和弦填充。
    """
    if not notes:
        return []
    
    # 只取低音区 < C4(60) 
    low_notes = [n for n in notes if n['pitch'] < 60]
    if not low_notes:
        return []
    
    low_notes.sort(key=lambda n: (n['start'], n['pitch']))
    
    # 时间窗 0.1s
    t_min = min(n['start'] for n in low_notes)
    t_max = max(n['end'] for n in low_notes)
    window = 0.1
    n_win = int((t_max - t_min) / window) + 1
    
    bass_pitches = {}  # round(start,1) -> [pitch1, pitch2, ...]
    
    for i in range(n_win):
        ws = t_min + i * window
        we = ws + window
        
        active = [(n['pitch'], n['end'] - n['start'], n)
                  for n in low_notes if n['start'] < we and n['end'] > ws]
        if len(active) < 1:
            continue
        
        # 按音高排序，取最低的1~2个
        active.sort(key=lambda x: (x[0], -x[1]))
        top = active[:2]  # 最多两个低音
        
        for pitch, dur, n in top:
            tk = round(ws, 1)
            if tk not in bass_pitches:
                bass_pitches[tk] = []
            bass_pitches[tk].append((pitch, n['start'], n['end']))
    
    # 转为音符
    result = []
    sorted_keys = sorted(bass_pitches.keys())
    
    for key in sorted_keys:
        entries = bass_pitches[key]
        for pitch, start, end in entries:
            if end - start >= min_duration:
                # 与上一个同音高的合并
                if result and result[-1]['pitch'] == pitch and \
                   start - result[-1]['end'] <= 0.15:
                    result[-1]['end'] = max(result[-1]['end'], end)
                else:
                    result.append({
                        'start': start,
                        'end': end,
                        'pitch': pitch,
                        'velocity': 70
                    })
    
    return result


def write_multitrack_midi(track_notes: list, output_path: str,
                          tempo: float = 120.0,
                          programs: list = None) -> str:
    """写入多音轨 MIDI 文件"""
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    if programs is None:
        programs = [0] * len(track_notes)
    for i, notes in enumerate(track_notes):
        prog = programs[i] if i < len(programs) else 0
        instr = pretty_midi.Instrument(program=prog)
        for n in notes:
            if n['start'] < 0 or n['end'] <= n['start']:
                continue
            mn = pretty_midi.Note(
                velocity=n['velocity'], pitch=n['pitch'],
                start=n['start'], end=n['end']
            )
            instr.notes.append(mn)
        midi.instruments.append(instr)
    midi.write(output_path)
    return output_path
