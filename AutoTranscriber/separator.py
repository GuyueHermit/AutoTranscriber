"""音源分离模块 — 将歌曲分离为人声和伴奏轨道

支持输入格式：.wav, .mp3, .flac, .ogg, .m4a, .aac, .wma 等 librosa 支持的格式。
非 WAV 文件会自动转码为临时 WAV 再传给 Demucs 处理。
"""

import os
import sys
import subprocess
import tempfile
import numpy as np
import soundfile as sf


# Demucs 安装在 ONN conda 环境中
DEMUCS_PYTHON = r"C:\Users\kotsu\miniconda3\envs\ONN\python.exe"

# librosa 支持的音频格式（通过 audioread + soundfile 后端）
SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac',
                     '.wma', '.aiff', '.au', '.raw', '.opus'}


def get_audio_format(path: str) -> str:
    """获取文件扩展名（小写）"""
    _, ext = os.path.splitext(path)
    return ext.lower()


def needs_conversion(path: str) -> bool:
    """判断是否需要先转换为 WAV 再传给 Demucs"""
    ext = get_audio_format(path)
    # Demucs (torchaudio + soundfile) 原生支持 wav 和 flac
    # 其他格式（mp3, m4a, ogg 等）需要 ffmpeg 或预转换
    if ext in {'.wav'}:
        return False
    return True


def convert_to_wav(input_path: str, output_dir: str = None,
                   target_sr: int = 44100) -> str:
    """
    将各种音频格式统一转换为 WAV 文件。

    使用 librosa 加载（支持 mp3/flac/ogg/m4a 等所有常见格式）
    再用 soundfile 写出标准 WAV。

    Parameters
    ----------
    input_path : str
        输入音频路径（任意格式）
    output_dir : str
        输出目录，None 则使用临时目录
    target_sr : int
        目标采样率，默认 44100（Demucs 推荐）

    Returns
    -------
    wav_path : str
        转换后的 WAV 文件路径
    """
    import librosa

    ext = get_audio_format(input_path)
    basename = os.path.splitext(os.path.basename(input_path))[0]

    print(f"  [转码] 格式检测: {ext} → 转码为 WAV")

    # 用 librosa 加载（支持所有常见格式）
    y, sr = librosa.load(input_path, sr=target_sr, mono=False)

    # 确定输出路径
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        wav_path = os.path.join(output_dir, f"{basename}_converted.wav")
    else:
        # 用临时文件
        tmp_dir = tempfile.mkdtemp(prefix="autotranscriber_")
        wav_path = os.path.join(tmp_dir, f"{basename}.wav")

    # 写出标准 WAV
    sf.write(wav_path, y.T if y.ndim > 1 else y, sr)

    input_size = os.path.getsize(input_path) / (1024 * 1024)
    output_size = os.path.getsize(wav_path) / (1024 * 1024)

    print(f"  [转码] 完成: {os.path.basename(input_path)} ({input_size:.1f}MB)")
    print(f"         → {os.path.basename(wav_path)} ({output_size:.1f}MB, {sr}Hz, {y.shape[-1]/sr:.1f}s)")

    return wav_path


def separate_audio(input_path: str, output_dir: str = None,
                   model: str = "htdemucs",
                   device: str = "cpu") -> dict:
    """
    使用 Demucs 将音频分离为人声和伴奏。

    自动处理格式转换：若输入非 WAV，先用 librosa 转码为 WAV 再处理。
    转码的临时文件在流程结束后自动清理。

    Parameters
    ----------
    input_path : str
        输入音频文件路径（支持 .wav, .mp3, .flac, .ogg, .m4a, .aac 等）
    output_dir : str
        分离结果输出目录，None 则在输入文件同目录下创建 separated/
    model : str
        Demucs 模型名 (htdemucs, htdemucs_ft, mdx_q, mdx_extra)
    device : str
        运行设备 (cpu/cuda)

    Returns
    -------
    paths : dict
        {
            'vocals': str,      # 人声 WAV 路径
            'no_vocals': str,   # 伴奏 WAV 路径
            'drums': str,       # 鼓（如可用）
            'bass': str,        # 贝斯（如可用）
            'other': str,       # 其他乐器（如可用）
            'mixture': str,     # 原始输入路径
        }
    """
    input_path = os.path.abspath(input_path)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    ext = get_audio_format(input_path)
    if ext not in SUPPORTED_FORMATS:
        print(f"  [分离] 警告: 格式 {ext} 未经验证，可能无法加载")
        print(f"  [分离] 支持的格式: {', '.join(sorted(SUPPORTED_FORMATS))}")

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(input_path), "separated")

    os.makedirs(output_dir, exist_ok=True)

    # ---- 格式转换：非 WAV → WAV ----
    temp_wav = None
    demucs_input = input_path

    if needs_conversion(input_path):
        temp_wav = convert_to_wav(input_path, output_dir)
        demucs_input = temp_wav

    # ---- 运行 Demucs ----
    cmd = [
        DEMUCS_PYTHON, "-m", "demucs",
        "--two-stems", "vocals",
        "-o", output_dir,
        "-n", model,
        "--device", device if device == "cuda" and __check_cuda() else "cpu",
        demucs_input
    ]

    print(f"  [分离] 运行 Demucs... (模型={model}, 设备=cpu)")
    print(f"  [分离] 这可能需要几分钟，请耐心等待")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600
        )
    except subprocess.TimeoutExpired:
        print(f"  [分离] 错误: Demucs 处理超时（10分钟）")
        raise

    # ---- 定位输出 ----
    # Demucs 输出结构: {output_dir}/{model}/{wav_basename}/vocals.wav
    demucs_basename = os.path.splitext(os.path.basename(demucs_input))[0]
    stem_dir = os.path.join(output_dir, model, demucs_basename)

    paths = {
        'vocals': os.path.join(stem_dir, 'vocals.wav'),
        'no_vocals': os.path.join(stem_dir, 'no_vocals.wav'),
        'mixture': input_path,
    }

    for stem in ['drums', 'bass', 'other']:
        stem_path = os.path.join(stem_dir, f'{stem}.wav')
        if os.path.exists(stem_path):
            paths[stem] = stem_path

    # ---- 验证结果 + 后备方案 ----
    demucs_ok = all(os.path.exists(paths.get(k, '')) for k in ['vocals', 'no_vocals'])

    if not demucs_ok:
        if result.returncode != 0:
            print(f"  [分离] Demucs 返回错误码 {result.returncode}")
            print(f"  [分离] stderr: {result.stderr[-300:]}")
        print(f"  [分离] 找不到分离输出，使用 HPSS 后备方案")
        hpss_paths = _hpss_separate(input_path, output_dir)
        paths.update(hpss_paths)

    # ---- 清理临时 WAV ----
    if temp_wav and os.path.exists(temp_wav):
        try:
            os.remove(temp_wav)
            # 清理临时目录（如果是单独创建的）
            tmp_dir = os.path.dirname(temp_wav)
            if os.path.basename(tmp_dir).startswith("autotranscriber_"):
                os.rmdir(tmp_dir)
        except Exception:
            pass

    return paths


def __check_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _hpss_separate(input_path: str, output_dir: str) -> dict:
    """HPSS 后备分离"""
    print("  [分离] 使用 HPSS 进行基本分离（质量不如 Demucs）")
    import librosa
    from scipy.signal import butter, sosfilt

    y, sr = librosa.load(input_path, sr=22050, mono=True)

    harmonic, percussive = librosa.effects.hpss(y)

    # 人声粗略估计：带通滤波 80~2000Hz
    sos = butter(4, [80 / (sr / 2), 2000 / (sr / 2)], btype='band', output='sos')
    vocals_rough = sosfilt(sos, harmonic)
    accompaniment = harmonic - vocals_rough + percussive * 0.5

    for sig in [vocals_rough, accompaniment]:
        max_val = np.max(np.abs(sig))
        if max_val > 0:
            sig /= max_val

    basename = os.path.splitext(os.path.basename(input_path))[0]
    hpss_dir = os.path.join(output_dir, "hpss", basename)
    os.makedirs(hpss_dir, exist_ok=True)

    vocals_path = os.path.join(hpss_dir, "vocals.wav")
    no_vocals_path = os.path.join(hpss_dir, "no_vocals.wav")

    sf.write(vocals_path, vocals_rough, sr)
    sf.write(no_vocals_path, accompaniment, sr)

    return {'vocals': vocals_path, 'no_vocals': no_vocals_path}


def has_demucs() -> bool:
    if not os.path.exists(DEMUCS_PYTHON):
        return False
    try:
        result = subprocess.run(
            [DEMUCS_PYTHON, "-m", "demucs", "--help"],
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0
    except Exception:
        return False
