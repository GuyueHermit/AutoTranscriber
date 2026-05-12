"""MIDI 转 PDF 乐谱模块

支持两种方式：
1. MuseScore（专业五线谱，推荐）— 需安装 MuseScore
2. matplotlib 钢琴卷帘谱（兜底，无需额外安装）
"""

import os
import subprocess
import sys
import tempfile
import numpy as np
import pretty_midi


def find_musescore() -> str:
    """查找系统中 MuseScore 可执行文件路径"""
    # 常见安装路径
    candidates = [
        # MuseScore 4
        os.path.expandvars(r"%ProgramFiles%\MuseScore 4\bin\MuseScore4.exe"),
        os.path.expandvars(r"%ProgramFiles%\MuseScore 4\MuseScore4.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\MuseScore 4\bin\MuseScore4.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\MuseScore 4\bin\MuseScore4.exe"),
        # MuseScore 3
        os.path.expandvars(r"%ProgramFiles%\MuseScore 3\bin\MuseScore3.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\MuseScore 3\bin\MuseScore3.exe"),
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\MuseScore 3\bin\MuseScore3.exe"),
        # 也可能直接叫 musescore
        os.path.expandvars(r"%ProgramFiles%\MuseScore\bin\MuseScore.exe"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    # 最后试试 PATH 里找
    try:
        for name in ["MuseScore4.exe", "MuseScore3.exe", "musescore3.exe", "MuseScore.exe"]:
            result = subprocess.run(
                ["where", name],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]
    except Exception:
        pass

    return None


def midi_to_pdf_musescore(midi_path: str, pdf_path: str,
                          musescore_path: str = None) -> bool:
    """
    使用 MuseScore CLI 将 MIDI 转为 PDF 五线谱。
    效果最好，能生成标准的五线谱排版。

    Parameters
    ----------
    midi_path : str
        输入的 .mid 文件路径
    pdf_path : str
        输出的 .pdf 文件路径
    musescore_path : str, optional
        MuseScore 可执行文件路径，自动查找

    Returns
    -------
    success : bool
        转换是否成功
    """
    if musescore_path is None:
        musescore_path = find_musescore()

    if not musescore_path or not os.path.exists(musescore_path):
        return False

    cmd = [
        musescore_path,
        os.path.abspath(midi_path),
        "-o",
        os.path.abspath(pdf_path)
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.returncode == 0 and os.path.exists(pdf_path)
    except Exception as e:
        print(f"  [PDF] MuseScore 转换失败: {e}")
        return False


def midi_to_pdf_pianoroll(midi_path: str, pdf_path: str,
                          track_names: list = None) -> str:
    """
    使用 matplotlib 将 MIDI 渲染为钢琴卷帘谱 PDF。
    无需额外软件，但呈现的是钢琴卷帘图而非五线谱。

    Parameters
    ----------
    midi_path : str
        输入的 .mid 文件路径
    pdf_path : str
        输出的 .pdf 文件路径
    track_names : list, optional
        各音轨名称

    Returns
    -------
    pdf_path : str
        生成的 PDF 路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    midi = pretty_midi.PrettyMIDI(midi_path)

    # 确定总时长
    total_time = max(
        (n.end for instr in midi.instruments for n in instr.notes),
        default=10.0
    )

    n_tracks = len(midi.instruments)
    if track_names is None:
        import pretty_midi as pm
        track_names = [
            pm.program_to_instrument_name(instr.program) if hasattr(pm, 'program_to_instrument_name') else f"Track {i}"
            for i, instr in enumerate(midi.instruments)
        ]

    # 计算布局：每轨一个子图
    fig, axes = plt.subplots(n_tracks, 1, figsize=(14, max(4, n_tracks * 3)))
    if n_tracks == 1:
        axes = [axes]

    # 标题（尝试支持中文）
    title_text = os.path.splitext(os.path.basename(midi_path))[0]
    # 用英文作为后备标题
    try:
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
    except Exception:
        fig.suptitle("Piano Roll - " + title_text.encode('ascii', errors='replace').decode(),
                     fontsize=14, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, max(n_tracks, 3)))

    for idx, (instr, ax) in enumerate(zip(midi.instruments, axes)):
        notes = instr.notes
        if not notes:
            ax.text(0.5, 0.5, "(无音符)", ha='center', va='center',
                    transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{track_names[idx]}")
            continue

        pitches = [n.pitch for n in notes]
        min_pitch = max(0, min(pitches) - 3)
        max_pitch = min(127, max(pitches) + 3)

        for note in notes:
            rect = Rectangle(
                (note.start, note.pitch - 0.4),
                note.end - note.start,
                0.8,
                facecolor=colors[idx],
                edgecolor='black',
                linewidth=0.3,
                alpha=0.85
            )
            ax.add_patch(rect)

        # 坐标轴
        ax.set_xlim(0, total_time + 0.5)
        ax.set_ylim(min_pitch, max_pitch)
        ax.set_xlabel("时间 (秒)" if idx == n_tracks - 1 else "")
        ax.set_ylabel("MIDI 音高")
        ax.set_title(f"{track_names[idx]} ({len(notes)} 个音符)")
        ax.grid(True, alpha=0.3, linestyle='--')

        # 在右侧标注音名
        from matplotlib.ticker import FuncFormatter
        def pitch_label(p, _):
            names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            octave = int(p // 12) - 1
            return f"{names[int(p % 12)]}{octave}"
        ax.yaxis.set_major_formatter(FuncFormatter(pitch_label))

    plt.tight_layout()
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return pdf_path


def midi_to_pdf(midi_path: str, pdf_path: str = None,
                force_pianoroll: bool = False) -> str:
    """
    MIDI → PDF 乐谱转换（自动选择最佳方式）。

    优先使用 MuseScore 生成专业五线谱 PDF，
    若不可用则用 matplotlib 生成钢琴卷帘谱 PDF。

    Parameters
    ----------
    midi_path : str
        输入的 .mid 文件路径
    pdf_path : str, optional
        输出的 .pdf 文件路径。None 则自动生成在同目录
    force_pianoroll : bool
        强制使用钢琴卷帘模式（即使有 MuseScore）

    Returns
    -------
    pdf_path : str
        生成的 PDF 文件路径
    """
    if not os.path.exists(midi_path):
        raise FileNotFoundError(f"MIDI 文件不存在: {midi_path}")

    if pdf_path is None:
        base = os.path.splitext(midi_path)[0]
        pdf_path = f"{base}.pdf"

    os.makedirs(os.path.dirname(pdf_path) or '.', exist_ok=True)

    # 方式一：MuseScore 专业五线谱
    if not force_pianoroll:
        musescore_path = find_musescore()
        if musescore_path:
            print(f"  [PDF] 使用 MuseScore 生成五线谱 PDF...")
            success = midi_to_pdf_musescore(midi_path, pdf_path, musescore_path)
            if success:
                print(f"  [PDF] 五线谱 PDF 已生成: {pdf_path}")
                return pdf_path
            else:
                print(f"  [PDF] MuseScore 转换失败，切换为钢琴卷帘模式")

    # 方式二：matplotlib 钢琴卷帘（兜底）
    print(f"  [PDF] 使用 matplotlib 生成钢琴卷帘谱 PDF...")
    pdf_path = midi_to_pdf_pianoroll(midi_path, pdf_path)
    print(f"  [PDF] 钢琴卷帘谱 PDF 已生成: {pdf_path}")
    return pdf_path


def install_musescore_guide() -> str:
    """返回安装 MuseScore 的指引"""
    return """
如果你想获得专业的五线谱 PDF（而不是钢琴卷帘图），
请安装免费开源的 MuseScore 打谱软件：

  方法 1（推荐，自动安装）:
    在终端运行: winget install MuseScore.MuseScore

  方法 2（手动安装）:
    访问 https://musescore.org 下载安装

安装后重新运行即可自动使用 MuseScore 生成五线谱 PDF。
"""
