#!/usr/bin/env python3
"""
AutoTranscriber v1.0 — 音频自动扒谱工具 🎵

只需三步：
1. 放一首歌或一段音频进来
2. 告诉方宜要不要分离人声
3. 拿到 MIDI 乐谱和 PDF 谱子

用法:
    python main.py -i 歌曲.mp3                    # 直接扒谱
    python main.py -i 歌曲.mp3 --pdf              # 扒谱 + 导出 PDF 谱子
    python main.py -i 歌曲.mp3 --separate --pdf   # 分离人声伴奏 + 扒谱 + PDF
    python main.py -i 歌曲.mp3 --pdf --view       # 扒谱 + PDF + 自动打开
"""

import argparse
import sys
import os
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from AutoTranscriber import (
    load_audio,
    compute_cqt,
    compute_spectral_flux,
    detect_onsets,
    estimate_pitches,
    estimate_pitches_onset_driven,
    estimate_vocal_pitch,
    track_notes,
    track_notes_onset_driven,
    perceptual_filter,
    write_midi,
    write_multitrack_midi,
    merge_tracks_to_piano,
    preprocess,
    has_demucs,
    separate_audio,
    midi_to_pdf,
    find_musescore,
    install_musescore_guide,
    estimate_vocal_pitch_crepe,
    has_crepe,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AutoTranscriber v1.0 — 音频自动扒谱工具 🎵\n"
                    "输入一段音频，自动生成 MIDI 乐谱和 PDF 谱子。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例（简单用法）:
  python main.py -i 歌曲.mp3                 扒谱，生成 .mid 文件
  python main.py -i 歌曲.mp3 --pdf           扒谱 + 生成 PDF 谱子
  python main.py -i 歌曲.mp3 --separate --pdf 分离人声+伴奏，双轨扒谱+PDF

高级用法:
  python main.py -i 音频.wav -o 输出.mid --n_peaks 6 --onset_threshold 0.2
        """
    )

    # ---- 核心参数（简单） ----
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入音频文件（.mp3/.wav/.flac/.m4a/... 都支持）')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出 MIDI 文件路径（不填则自动命名）')

    # ---- 一键功能（让小白也能用） ----
    parser.add_argument('--pdf', action='store_true',
                        help='📄 同时生成 PDF 谱子（方便直接看/打印）')
    parser.add_argument('--separate', '-s', action='store_true',
                        help='🎤 分离人声+伴奏再扒谱（适合歌曲）')
    parser.add_argument('--piano', action='store_true',
                        help='🎹 钢琴曲模式（优化节奏和音准，去除杂音）')
    parser.add_argument('--perceptual', '-p', action='store_true',
                        help='🧠 感知模式：基于起始点检测+振幅追踪+听感滤波（解决余音干扰）')
    parser.add_argument('--solo', action='store_true',
                        help='🎹 合并为单人钢琴谱（一人弹，默认四手联弹）')
    parser.add_argument('--view', action='store_true',
                        help='👁 生成后自动打开 PDF/MIDI')

    # ---- 高级参数（专业人士用） ----
    parser.add_argument('--track', type=str, default='both',
                        choices=['both', 'vocals', 'accompaniment'],
                        help='分离后只扒哪一轨 (both/vocals/accompaniment)')
    parser.add_argument('--demucs_model', type=str, default='htdemucs',
                        help='分离模型 (默认: htdemucs)')
    parser.add_argument('--n_peaks', type=int, default=5,
                        help='最大同时音符数 (和弦5-8, 人声1-2)')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='帧移 (越小精度越高)')
    parser.add_argument('--onset_threshold', type=float, default=0.3,
                        help='起始检测灵敏度 (0~1)')
    parser.add_argument('--pitch_threshold', type=float, default=0.1,
                        help='音高检测阈值')
    parser.add_argument('--min_note_duration', type=int, default=4,
                        help='最小音符时长（过滤噪声）')
    parser.add_argument('--bins_per_octave', type=int, default=36,
                        help='频谱精度 (12=半音, 36=1/3半音)')
    parser.add_argument('--simplify', type=int, default=0,
                        help='🎯 音符精简: 0=关闭, 3=强烈(推荐钢琴), 5=轻度, 2=极致精简')
    parser.add_argument('--tempo', type=float, default=120.0,
                        help='MIDI 速度 BPM')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')

    return parser.parse_args()


def print_progress(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def auto_output_path(input_path: str, ext: str = '.mid') -> str:
    """自动生成输出路径"""
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(os.path.dirname(input_path) or '.', f"{base}{ext}")


def transcribe_file(audio_path: str, args, label: str = None,
                    use_pyin: bool = False, piano_mode: bool = False) -> list:
    """对单个音频执行完整扒谱
    
    use_pyin=True: 使用 pYIN 算法（适合人声旋律）
    piano_mode=True: 钢琴曲优化模式（节奏更准、杂音更少）
    """
    tag = f"[{label}] " if label else ""

    y, sr = load_audio(audio_path, sr=args.sr)
    y = preprocess(y, sr)
    duration = len(y) / sr
    print(f"      {tag}时长: {duration:.1f}秒")

    if use_pyin:
        print(f"      {tag}使用 pYIN 算法检测人声音高...")
        notes = estimate_vocal_pitch(
            y, sr,
            hop_length=args.hop_length,
            min_duration_frames=args.min_note_duration
        )
        print(f"      {tag}检测到 {len(notes)} 个旋律音符")
        return notes

    # ---- 钢琴模式参数 ----
    if piano_mode:
        n_peaks = 2
        onset_th = 0.15
        pitch_th = 0.2
        hop = 256  # 更高时间分辨率
    else:
        n_peaks = args.n_peaks
        onset_th = args.onset_threshold
        pitch_th = args.pitch_threshold
        hop = args.hop_length

    # CQT 频谱
    cqt, times, freqs = compute_cqt(
        y, sr, hop_length=hop,
        fmin=65.41, fmax=2093.0,
        bins_per_octave=args.bins_per_octave
    )

    # 起始检测（钢琴模式用更低阈值）
    flux = compute_spectral_flux(cqt)
    onset_frames, onset_times = detect_onsets(
        flux, sr, hop_length=hop,
        threshold=onset_th
    )
    
    if len(onset_frames) > 0:
        print(f"      {tag}检测到 {len(onset_frames)} 个节奏起始点")
    else:
        print(f"      {tag}未检测到起始点，使用连续追踪")

    # 多音高估计
    frame_notes = estimate_pitches(
        cqt, freqs, times, sr,
        hop_length=hop,
        n_peaks=n_peaks,
        threshold_factor=pitch_th
    )

    # 对钢琴模式做中值滤波（去除单帧毛刺）
    if piano_mode:
        frame_notes = _median_filter_frames(frame_notes, window=3)

    # 音符追踪
    notes = track_notes(
        frame_notes, onset_frames, onset_times,
        times, sr, hop_length=hop,
        min_note_duration=args.min_note_duration,
        velocity_scale=80.0
    )

    # 精简化
    if args.simplify > 0 or piano_mode:
        sp = args.simplify if args.simplify > 0 else 2
        before = len(notes)
        notes = _simplify_notes(notes, max_per_beat=sp)
        print(f"      {tag}精简: {before} → {len(notes)} 个音符")

    return notes


def transcribe_file_perceptual(audio_path: str, args, label: str = None) -> list:
    """
    [新方法] 基于起始点的感知式扒谱。

    与旧方法的区别：
    1. 只在起始点检测新进入的音（非逐帧检测）
    2. 用振幅衰减追踪每个音的持续时间
    3. 用感知滤波去除离群音和谐波
    """
    tag = f"[{label}] " if label else ""

    y, sr = load_audio(audio_path, sr=args.sr)
    y = preprocess(y, sr)
    duration = len(y) / sr
    print(f"      {tag}时长: {duration:.1f}秒")

    hop = args.hop_length

    # CQT 频谱
    print(f"      {tag}计算 CQT 频谱...")
    cqt, times, freqs = compute_cqt(
        y, sr, hop_length=hop,
        fmin=65.41, fmax=2093.0,
        bins_per_octave=args.bins_per_octave
    )

    # 起始检测
    print(f"      {tag}检测音符起始点...")
    flux = compute_spectral_flux(cqt)
    onset_frames, onset_times = detect_onsets(
        flux, sr, hop_length=hop,
        threshold=args.onset_threshold
    )
    print(f"      {tag}找到 {len(onset_frames)} 个起始点")

    # 只在起始点检测新进入的音
    print(f"      {tag}基于起始点的音高检测...")
    onset_notes = estimate_pitches_onset_driven(
        cqt, freqs, times,
        onset_frames, onset_times,
        sr, hop_length=hop,
        n_peaks=args.n_peaks,
        threshold_factor=args.pitch_threshold
    )
    print(f"      {tag}检测到 {len(onset_notes)} 个起始音")

    # 追踪每个音的持续时间（振幅衰减法）
    print(f"      {tag}追踪音符持续时间...")
    notes = track_notes_onset_driven(
        onset_notes, cqt, freqs, times,
        sr, hop_length=hop,
        decay_ratio=0.25,
        snr_threshold=0.3
    )
    print(f"      {tag}追踪到 {len(notes)} 个音符")

    # 感知滤波
    print(f"      {tag}感知滤波（去离群音+谐波+噪声）...")
    before = len(notes)
    notes = perceptual_filter(
        notes,
        outlier_semitones=12,
        min_duration=0.06,
        harmonic_check=True,
        max_simultaneous=6,
        max_notes_per_beat=8
    )
    print(f"      {tag}滤波: {before} → {len(notes)} 个音符")

    return notes


def _median_filter_frames(frame_notes: list, window: int = 3) -> list:
    """对逐帧音高做中值滤波，去除单帧毛刺"""
    import numpy as np
    if not frame_notes:
        return frame_notes
    n_frames = len(frame_notes)
    result = []
    for i in range(n_frames):
        notes_now = frame_notes[i]
        if not notes_now:
            result.append([])
            continue
        
        # 对每个当前帧的音符，看邻居帧是否有相同音高
        # 如果有邻居支持则保留，否则移除
        filtered = []
        for note in notes_now:
            pitch = note['pitch']
            support = 0
            for j in range(max(0, i-window), min(n_frames, i+window+1)):
                if j != i:
                    for n in frame_notes[j]:
                        if abs(n['pitch'] - pitch) <= 1:
                            support += 1
                            break
            # 至少有一个邻居支持才保留
            if support >= 1:
                filtered.append(note)
        
        if not filtered:
            filtered = notes_now  # 如果没有被过滤的，至少保留最强的
            if len(filtered) > 1:
                filtered = [max(filtered, key=lambda x: x['amplitude'])]
        
        result.append(filtered)
    return result


def _simplify_notes(notes: list, max_per_beat: int = 3) -> list:
    """
    音符精简化：去毛刺+合并+选音。
    
    三步：
    1. 去毛刺：删除极短的孤立音符（< 0.08s）
    2. 合并：相近音高 + 短间隔 → 合并为长音
    3. 选音：每 0.15s 窗只留最重要的音
    """
    if not notes:
        return []

    notes = [dict(n) for n in notes]
    notes.sort(key=lambda n: (n['start'], n['pitch']))

    # ---- 1. 去毛刺 ----
    notes = [n for n in notes if n['end'] - n['start'] >= 0.08]

    # ---- 2. 合并相邻同音高/近音高 ----
    merged = []
    for n in notes:
        if merged:
            last = merged[-1]
            gap = n['start'] - last['end']
            pdiff = abs(n['pitch'] - last['pitch'])
            # 同音高且间隔 < 0.2s → 合并
            if pdiff == 0 and gap < 0.2:
                last['end'] = max(last['end'], n['end'])
                continue
            # 音高差 ≤ 1 且间隔 < 0.1s → 取更长的那个音高
            if pdiff <= 1 and gap < 0.1:
                if n['end'] - n['start'] > last['end'] - last['start']:
                    last['pitch'] = n['pitch']
                    last['end'] = n['end']
                continue
        merged.append(n)
    
    # ---- 3. 时间窗选音 ----
    if max_per_beat <= 0:
        return merged
    
    t_min = min(n['start'] for n in merged)
    t_max = max(n['end'] for n in merged)
    window = 0.15
    n_win = int((t_max - t_min) / window) + 1
    kept = set()
    
    for i in range(n_win):
        ws = t_min + i * window
        we = ws + window
        active = [(n, n['end'] - n['start']) for n in merged 
                  if n['start'] < we and n['end'] > ws and id(n) not in kept]
        if not active:
            continue
        # 评分：高音旋律 > 低音骨干 > 中音填充
        active.sort(key=lambda x: (
            3 if x[0]['pitch'] >= 67 else 2 if x[0]['pitch'] < 48 else 1,
            x[1],  # 时长
            x[0]['pitch'] if x[0]['pitch'] >= 67 else -x[0]['pitch']
        ), reverse=True)
        for n, _ in active[:max_per_beat]:
            kept.add(id(n))
    
    result = [n for n in merged if id(n) in kept]
    result.sort(key=lambda n: (n['start'], -n['pitch']))
    return result


def open_file(filepath: str):
    """自动打开文件（跨平台）"""
    try:
        if sys.platform == 'win32':
            os.startfile(filepath)
        elif sys.platform == 'darwin':
            subprocess.run(['open', filepath])
        else:
            subprocess.run(['xdg-open', filepath])
    except Exception as e:
        print(f"  [提示] 无法自动打开文件，请手动打开: {filepath}")


def main():
    args = parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"❌ 错误: 文件不存在 → {input_path}")
        sys.exit(1)

    # 自动输出路径
    if args.output is None:
        args.output = auto_output_path(input_path, '.mid')

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # =========================================================
    # 欢迎信息
    # =========================================================
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║     AutoTranscriber v1.0 — 自动扒谱     ║")
    print("  ╚══════════════════════════════════════════╝")
    print()
    print(f"  📂 输入: {os.path.basename(input_path)}")
    print(f"  📝 输出: {os.path.basename(args.output)}")

    features = []
    if args.piano:
        features.append("🎹 钢琴模式")
    if args.separate:
        features.append("🎤 分离人声+伴奏")
    if args.solo:
        features.append("🎹 单人钢琴")
    if args.pdf:
        features.append("📄 生成 PDF 谱子")
    if args.view:
        features.append("👁 自动打开")
    if features:
        print(f"  ✨ 功能: {' | '.join(features)}")
    print()

    # =========================================================
    # 模式 A: 音源分离 + 扒谱
    # =========================================================
    if args.separate:
        if not has_demucs():
            print("❌ Demucs 不可用，无法分离。请先安装:")
            print("   conda activate ONN && pip install demucs")
            sys.exit(1)

        print_progress("阶段 1/3: 音源分离 (Demucs)...")
        print("      正在分离人声和伴奏... (约需 1-5 分钟)")
        separated = separate_audio(input_path, model=args.demucs_model)

        track_map = {
            'vocals': [('vocals', '人声', 0)],
            'accompaniment': [('no_vocals', '伴奏', 1)],
            'both': [('no_vocals', '伴奏', 1), ('vocals', '人声', 0)],
        }

        tracks_to_process = track_map[args.track]
        all_track_notes = []
        programs = []

        print_progress("阶段 2/3: 扒谱...")
        for stem_key, stem_label, program in tracks_to_process:
            stem_path = separated.get(stem_key)
            if not stem_path or not os.path.exists(stem_path):
                print(f"      跳过 {stem_label}（文件不存在）")
                continue
            print(f"\n      ── {stem_label} ──")
            
            if stem_label == '人声' and has_crepe():
                print(f"      🧠 人声模式: 使用 CREPE 深度学习音高检测")
                notes = estimate_vocal_pitch_crepe(stem_path)
            else:
                # 伴奏用 CQT（支持钢琴模式）
                notes = transcribe_file(stem_path, args, label=stem_label,
                                        piano_mode=args.piano)
            
            print(f"      ✅ 找到 {len(notes)} 个音符")
            all_track_notes.append(notes)
            programs.append(program)

        if not all_track_notes:
            print("❌ 没有可扒的音轨")
            sys.exit(1)

        print_progress("阶段 3/3: 生成 MIDI...")

        if args.solo and len(all_track_notes) > 1:
            # 合并为单人钢琴谱
            print(f"      🎹 合并为单人钢琴谱中...")
            # 将轨道信息打包（含类型）
            # tracks_to_process: [('no_vocals','伴奏',1), ('vocals','人声',0)]
            track_info = []
            for i, (_, track_type, _) in enumerate(tracks_to_process):
                if i < len(all_track_notes):
                    ttype = 'vocal' if track_type == '人声' else 'accompaniment'
                    track_info.append((all_track_notes[i], ttype))
            merged = merge_tracks_to_piano(track_info, max_simultaneous=5)
            output_path = write_midi(merged, args.output,
                                     tempo=args.tempo, program=0)
            total = len(merged)
            print(f"      ✅ 单人钢琴谱已生成: {output_path}")
            print(f"         (合并前 {sum(len(n) for n in all_track_notes)} 个音符 → 合并后 {total} 个音符)")
        elif len(all_track_notes) == 1:
            output_path = write_midi(all_track_notes[0], args.output,
                                     tempo=args.tempo, program=programs[0])
            total = len(all_track_notes[0])
        else:
            output_path = write_multitrack_midi(all_track_notes, args.output,
                                                tempo=args.tempo, programs=programs)
            total = sum(len(n) for n in all_track_notes)
            print(f"      ✅ 四手联弹谱已生成: {output_path} (双轨, {total} 个音符)")

    # =========================================================
    # 模式 B: 直接扒谱
    # =========================================================
    else:
        if args.perceptual:
            print_progress("🧠 感知模式扒谱（基于起始点+振幅追踪+听感滤波）...")
            notes = transcribe_file_perceptual(input_path, args)
        else:
            print_progress("扒谱中...")
            notes = transcribe_file(input_path, args, piano_mode=args.piano)
        output_path = write_midi(notes, args.output,
                                 tempo=args.tempo, program=0)
        print(f"      ✅ MIDI 已保存: {output_path} ({len(notes)} 个音符)")

    # =========================================================
    # PDF 生成（可选）
    # =========================================================
    if args.pdf:
        print()
        print_progress("生成 PDF 谱子...")

        pdf_path = os.path.splitext(args.output)[0] + '.pdf'

        # 检测 MuseScore
        musescore = find_musescore()
        if musescore:
            print(f"  [PDF] 检测到 MuseScore，生成专业五线谱...")
        else:
            print(f"  [PDF] 未检测到 MuseScore，使用钢琴卷帘图")
            print(f"  [PDF] 如需五线谱，安装 MuseScore 后效果更好")
            print(f"  [PDF] 安装方法: winget install MuseScore.MuseScore")

        pdf_path = midi_to_pdf(output_path, pdf_path)
        pdf_size = os.path.getsize(pdf_path) / 1024
        print(f"      ✅ PDF 已保存: {pdf_path} ({pdf_size:.0f}KB)")

        if args.view:
            print(f"      👁 正在打开...")
            open_file(pdf_path)

    # =========================================================
    # 自动打开 MIDI（可选）
    # =========================================================
    if args.view and not args.pdf:
        print(f"      👁 正在打开 MIDI...")
        open_file(output_path)

    # =========================================================
    # 完成
    # =========================================================
    print()
    print("  ╔══════════════════════════════════════════╗")
    print("  ║     ✅ 全部完成！                        ║")
    print("  ╚══════════════════════════════════════════╝")
    print()


if __name__ == '__main__':
    main()
