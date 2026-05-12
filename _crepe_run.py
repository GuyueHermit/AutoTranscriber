"""CREPE 音高检测 — 被主项目通过子进程调用"""
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
