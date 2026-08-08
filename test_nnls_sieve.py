"""验证 _harmonic_sieve 新版（NNLS + 预计算表）在合成和弦上的表现"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\kotsu\Desktop\CODE\pythonProject\AutoTranscriber")

from AutoTranscriber.spectral import compute_cqt
from AutoTranscriber.pitch_estimation import (
    _harmonic_sieve, build_harmonic_table, midi_to_name,
)

# ---------- 测试 1: C4+E4+G4 和弦（带谐波） ----------
sr = 22050
dur = 1.0
t = np.linspace(0, dur, int(sr * dur), endpoint=False)

y = np.zeros_like(t)
for f, a in [(261.63, 1.0), (329.63, 0.9), (392.00, 0.8)]:
    for h in range(1, 4):  # 3 次谐波
        y += a * (1.0 / h) * np.sin(2 * np.pi * f * h * t)
y /= np.max(np.abs(y))

cqt, times, freqs = compute_cqt(y, sr, hop_length=512)
frame_idx = cqt.shape[1] // 2
frame = cqt[:, frame_idx]

table = build_harmonic_table(freqs)
notes = _harmonic_sieve(frame, freqs, n_peaks=5, threshold_factor=0.05,
                        harmonic_table=table)

print("=== 测试1: C4+E4+G4 和弦（谐波重叠场景） ===")
for n in notes:
    print(f"  {midi_to_name(n['pitch']):>4s}  freq={n['frequency']:7.2f} Hz  amp={n['amplitude']:.3f}")

detected = {n['pitch'] for n in notes}
expected = {60, 64, 67}
hit = expected & detected
print(f"期望: C4 E4 G4 | 命中 {len(hit)}/3 -> {sorted(midi_to_name(p) for p in hit)}")
print("PASS" if len(hit) >= 2 else "FAIL")
print()

# ---------- 测试 2: 单音 C4（应只检测一个音符，不报泛音） ----------
y2 = np.zeros_like(t)
for h in range(1, 5):
    y2 += (1.0 / h) * np.sin(2 * np.pi * 261.63 * h * t)
y2 /= np.max(np.abs(y2))

cqt2, _, _ = compute_cqt(y2, sr, hop_length=512)
frame2 = cqt2[:, cqt2.shape[1] // 2]
notes2 = _harmonic_sieve(frame2, freqs, n_peaks=5, threshold_factor=0.05,
                         harmonic_table=table)

print("=== 测试2: 单音 C4（应只报 1 个音符） ===")
for n in notes2:
    print(f"  {midi_to_name(n['pitch']):>4s}  freq={n['frequency']:7.2f} Hz  amp={n['amplitude']:.3f}")
print(f"检测到 {len(notes2)} 个音符")
print("PASS" if len(notes2) == 1 else "FAIL")

# ---------- 测试 3: 八度双音 C3+C4（泛音 vs 真实音符） ----------
y3 = np.zeros_like(t)
for f, a in [(130.81, 1.0), (261.63, 0.9)]:
    for h in range(1, 4):
        y3 += a * (1.0 / h) * np.sin(2 * np.pi * f * h * t)
y3 /= np.max(np.abs(y3))

cqt3, _, _ = compute_cqt(y3, sr, hop_length=512)
frame3 = cqt3[:, cqt3.shape[1] // 2]
notes3 = _harmonic_sieve(frame3, freqs, n_peaks=4, threshold_factor=0.05,
                         harmonic_table=table)

print("\n=== 测试3: C3+C4 八度双音 ===")
for n in notes3:
    print(f"  {midi_to_name(n['pitch']):>4s}  freq={n['frequency']:7.2f} Hz  amp={n['amplitude']:.3f}")
det3 = {n['pitch'] for n in notes3}
print("PASS" if len(det3) >= 2 else "FAIL (八度音可能被谐波抑制)")

# ---------- 测试 4: suppress_ratio 可选抑制旋钮验证 ----------
print("\n=== 测试4: suppress_ratio 控制显式八度抑制 ===")
n1 = _harmonic_sieve(frame3, freqs, n_peaks=4, threshold_factor=0.05,
                     harmonic_table=table)  # 默认 None = 信任 NNLS
n2 = _harmonic_sieve(frame3, freqs, n_peaks=4, threshold_factor=0.05,
                     harmonic_table=table, suppress_ratio=1.3)  # 显式抑制
d1 = {n['pitch'] for n in n1}
d2 = {n['pitch'] for n in n2}
print(f"  默认(None): {[midi_to_name(n['pitch']) for n in n1]}")
print(f"  显式(1.3):  {[midi_to_name(n['pitch']) for n in n2]}")
print("  默认保留 C3+C4 (真实八度音):", {48, 60} <= d1)
print("  显式抑制可关掉 C4:", 48 in d2 and 60 not in d2)
