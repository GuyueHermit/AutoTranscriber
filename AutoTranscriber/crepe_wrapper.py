"""CREPE 音高估计封装 — 通过子进程调用独立的 torchcrepe 环境"""

import os
import sys
import json
import subprocess
import tempfile
import numpy as np


# 专用虚拟环境 (python -m venv crepe_env)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREPE_PYTHON = os.path.join(BASE_DIR, "crepe_env", "Scripts", "python.exe")

# CREPE 辅助脚本路径
CREPE_SCRIPT = os.path.join(BASE_DIR, "_crepe_run.py")


def _ensure_crepe_script():
    """确保 CREPE 辅助脚本存在"""
    if not os.path.exists(CREPE_SCRIPT):
        with open(CREPE_SCRIPT, 'w', encoding='utf-8') as f:
            f.write(_CREPE_SCRIPT_CONTENT)
    return CREPE_SCRIPT


def estimate_vocal_pitch_crepe(audio_path: str,
                                hop_length: int = 320,
                                model: str = 'tiny',
                                energy_threshold: float = 0.02,
                                min_note_duration_frames: int = 5) -> list:
    """
    使用 CREPE 深度学习模型进行人声音高估计。

    通过子进程调用独立 Python 环境中的 torchcrepe。
    不依赖 ONN 或其他项目的环境。

    Parameters
    ----------
    audio_path : str
        人声 WAV 文件路径
    hop_length : int
        帧移
    model : str
        CREPE 模型容量 ('tiny' 或 'full')
    energy_threshold : float
        能量阈值，低于此值为静音
    min_note_duration_frames : int
        最小音符持续帧数

    Returns
    -------
    notes : list of dict
    """
    if not os.path.exists(CREPE_PYTHON):
        print(f"  [CREPE] 错误: 找不到 Python 环境")
        print(f"  [CREPE] 请先创建环境:")
        print(f"  [CREPE]   cd {BASE_DIR}")
        print(f"  [CREPE]   python -m venv crepe_env")
        print(f"  [CREPE]   crepe_env\\Scripts\\pip install torch torchaudio torchcrepe librosa soundfile pretty_midi")
        return []

    if not os.path.exists(audio_path):
        print(f"  [CREPE] 错误: 音频文件不存在: {audio_path}")
        return []

    script = _ensure_crepe_script()

    cmd = [
        CREPE_PYTHON, script,
        audio_path,
        "--hop_length", str(hop_length),
        "--model", model,
        "--energy_threshold", str(energy_threshold),
        "--min_duration", str(min_note_duration_frames),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            print(f"  [CREPE] 错误: {result.stderr[:300]}")
            return []

        notes = json.loads(result.stdout.strip())
        return notes

    except subprocess.TimeoutExpired:
        print(f"  [CREPE] 超时")
        return []
    except json.JSONDecodeError as e:
        print(f"  [CREPE] JSON 解析错误: {e}")
        print(f"  [CREPE] 输出: {result.stdout[:200]}")
        return []
    except Exception as e:
        print(f"  [CREPE] 错误: {e}")
        return []


def has_crepe() -> bool:
    """检查 CREPE 环境是否可用"""
    if not os.path.exists(CREPE_PYTHON):
        return False
    try:
        result = subprocess.run(
            [CREPE_PYTHON, "-c", "import torchcrepe"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False


# CREPE 辅助脚本（运行时自动写入）
_CREPE_SCRIPT_CONTENT = r'''"""CREPE 音高检测 — 被主项目通过子进程调用"""
import torchcrepe
import librosa
import numpy as np
import torch
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("audio_path")
parser.add_argument("--hop_length", type=int, default=320)
parser.add_argument("--model", default="tiny")
parser.add_argument("--energy_threshold", type=float, default=0.02)
parser.add_argument("--min_duration", type=int, default=5)
args = parser.parse_args()

SR = 16000

def freq_to_midi(f):
    return int(round(12 * np.log2(float(f) / 440.0) + 69))

audio, sr = librosa.load(args.audio_path, sr=SR)
hop = args.hop_length

# 帧能量
frame_energy = np.array([
    np.sum(audio[max(0,i):min(len(audio), i+hop)]**2) 
    for i in range(0, len(audio)-hop, hop)
])
if np.max(frame_energy) > 0:
    frame_energy = frame_energy / np.max(frame_energy)

# CREPE
audio_t = torch.from_numpy(audio).unsqueeze(0).float()
with torch.no_grad():
    f0 = torchcrepe.predict(audio_t, sr, hop_length=hop, batch_size=128, model=args.model)

f0_np = f0.cpu().numpy().flatten()
min_len = min(len(f0_np), len(frame_energy))
f0_np = f0_np[:min_len]
frame_energy = frame_energy[:min_len]

voiced = (f0_np > 0) & (frame_energy > args.energy_threshold)
pitch_seq = [freq_to_midi(f) if v else 0 for f, v in zip(f0_np, voiced)]

# 平滑
window = 3
smoothed = list(pitch_seq)
for i in range(window, len(pitch_seq) - window):
    if pitch_seq[i] > 0:
        nb = [pitch_seq[i+j] for j in range(-window, window+1) if pitch_seq[i+j] > 0]
        if nb:
            smoothed[i] = int(np.median(nb))

# 合并为音符
notes = []
cur_pitch = 0; cur_start = 0; cur_count = 0
times = np.arange(len(smoothed)) * hop / sr

for i, pitch in enumerate(smoothed):
    if pitch > 0:
        if cur_pitch == 0:
            cur_pitch = pitch; cur_start = i; cur_count = 1
        elif abs(pitch - cur_pitch) <= 1:
            cur_pitch = pitch; cur_count += 1
        else:
            if cur_count >= args.min_duration:
                notes.append({"start": float(times[cur_start]), "end": float(times[i]), "pitch": cur_pitch, "velocity": 80})
            cur_pitch = pitch; cur_start = i; cur_count = 1
    else:
        if cur_pitch > 0 and cur_count >= args.min_duration:
            notes.append({"start": float(times[cur_start]), "end": float(times[i]), "pitch": cur_pitch, "velocity": 80})
        cur_pitch = 0; cur_count = 0

if cur_pitch > 0 and cur_count >= args.min_duration:
    notes.append({"start": float(times[cur_start]), "end": float(times[-1]), "pitch": cur_pitch, "velocity": 80})

print(json.dumps(notes))
'''
