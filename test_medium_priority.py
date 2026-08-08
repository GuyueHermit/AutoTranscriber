"""验证中优先级改进：vibrato 轨迹追踪 + pYIN 动态力度"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\kotsu\Desktop\CODE\pythonProject\AutoTranscriber")
from AutoTranscriber.spectral import compute_cqt
from AutoTranscriber.pitch_estimation import (
    estimate_vocal_pitch, midi_to_name,
)
from AutoTranscriber.note_tracking import track_notes_onset_driven

sr = 22050
dur = 1.5
t = np.linspace(0, dur, int(sr * dur), endpoint=False)

# ---------- 测试 5: 滑音（portamento）频率轨迹追踪 ----------
# C4 线性滑到 G4（7 半音，0.8s）：固定频带锁死单 bin 会在滑音开始后
# 立即丢失能量；滑动窗口轨迹跟随能追踪到漂移保护边界。
dur2 = 0.8
t2 = np.linspace(0, dur2, int(sr * dur2), endpoint=False)
f_slide = 261.63 * 2 ** ((7.0 * t2 / dur2) / 12)
phase2 = 2 * np.pi * np.cumsum(f_slide) / sr
y_slide = np.sin(phase2)
y_slide /= np.max(np.abs(y_slide))

cqt_s, times_s, freqs_s = compute_cqt(y_slide, sr, hop_length=512)
on_s = [{
    'onset_frame': 8,
    'onset_time': times_s[8],
    'pitch': 60, 'frequency': 261.63,
    'amplitude': 0.8, 'snr': 5.0,
}]

n_fixed = track_notes_onset_driven(
    on_s, cqt_s, freqs_s, times_s, sr,
    track_semitones=0.0, max_drift_semitones=3.0)
n_track = track_notes_onset_driven(
    on_s, cqt_s, freqs_s, times_s, sr,
    track_semitones=1.0, max_drift_semitones=3.0)

def note_len(note):
    return note['end'] - note['start']

print("=== 测试5: 滑音 C4->G4 频率轨迹追踪 ===")
print(f"  固定窗口:  {len(n_fixed)} 个音符 {f'({note_len(n_fixed[0]):.2f}s)' if n_fixed else ''}")
print(f"  轨迹跟随:  {len(n_track)} 个音符 {f'({note_len(n_track[0]):.2f}s)' if n_track else ''}")
track_ok = n_track and (not n_fixed or note_len(n_track[0]) > note_len(n_fixed[0]))
print("  PASS" if track_ok else "  FAIL")

# ---------- 测试 6: pYIN 动态力度 ----------
# 0~0.6s 强 C4 (amp 1.0)，0.6~1.2s 弱 E4 (amp 0.2)
y2 = np.zeros_like(t)
seg1 = t < 0.6
seg2 = (t >= 0.6) & (t < 1.2)
y2[seg1] += 1.0 * np.sin(2 * np.pi * 261.63 * t[seg1])
y2[seg2] += 0.2 * np.sin(2 * np.pi * 329.63 * t[seg2])
y2 /= np.max(np.abs(y2))

notes2 = estimate_vocal_pitch(y2, sr, hop_length=512)
print("\n=== 测试6: pYIN 动态力度 ===")
for n in notes2:
    print(f"  {midi_to_name(n['pitch']):>4s}  {n['start']:.2f}s~{n['end']:.2f}s  velocity={n['velocity']}")
vels = [(n['pitch'], n['velocity']) for n in notes2]
strong = [v for p, v in vels if p == 60]
weak = [v for p, v in vels if p == 64]
print(f"  强音 C4 velocity: {strong}, 弱音 E4 velocity: {weak}")
print("  PASS" if strong and weak and strong[0] > weak[0] else "  FAIL")
