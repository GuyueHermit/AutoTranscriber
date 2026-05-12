# AutoTranscriber 🎵

音频自动扒谱工具 — 输入任意音频，输出 MIDI 乐谱文件。

支持**歌曲扒谱**：自动分离人声和伴奏，分别转为 MIDI 乐谱。

## 功能

- 🎤 **歌曲扒谱** — 使用 Demucs AI 音源分离，先分人声和伴奏再扒谱
- 🎸 **和弦识别** — 迭代谐波减法多音高估计，支持同时检测多个音符
- 🎹 **MIDI 输出** — 支持多音轨 MIDI（人声一轨+伴奏一轨）
- 🎵 基于 CQT（Constant-Q Transform）的频谱分析
- 📦 单音/旋律/和弦音频均可处理

## 安装

```bash
pip install -r requirements.txt
```

> 音源分离功能需额外安装 Demucs（已集成，自动可用）：
> ```
> conda create -n ONN python=3.10
> conda activate ONN
> pip install torch torchaudio
> pip install demucs soundfile av
> ```

## 使用方法

### 基本扒谱（乐器/单音轨）

```bash
python main.py -i 音频文件.wav -o 输出.mid
```

### 歌曲扒谱（分离人声+伴奏）

```bash
# 全自动：分离并扒取两轨
python main.py -i 歌曲.mp3 -o 歌曲.mid --separate

# 只想扒伴奏
python main.py -i 歌曲.mp3 -o 伴奏.mid --separate --track accompaniment

# 只想扒人声旋律（建议 n_peaks 调小）
python main.py -i 歌曲.mp3 -o 人声旋律.mid --separate --track vocals --n_peaks 2
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-i` | 输入音频路径（wav/mp3/flac 等） | 必填 |
| `-o` | 输出 MIDI 路径 | `output.mid` |
| `--separate` / `-s` | 开启音源分离模式 | 关 |
| `--track` | 分离模式扒哪一轨：`both`/`vocals`/`accompaniment` | `both` |
| `--demucs_model` | Demucs 分离模型 | `htdemucs` |
| `--n_peaks` | 每帧最大同时音符数（和弦设5-8，人声设1-2） | 5 |
| `--hop_length` | 帧移（越小时间分辨率越高） | 512 |
| `--bins_per_octave` | 每八度频带数（12=半音精度，36=1/3半音） | 36 |
| `--onset_threshold` | 起始检测灵敏度（0~1） | 0.3 |
| `--pitch_threshold` | 音高检测幅度阈值 | 0.1 |
| `--min_note_duration` | 最小音符时长（帧数，过滤短噪声） | 4 |
| `--tempo` | MIDI 速度 BPM | 120 |
| `--program` | MIDI 音色编号（单轨模式） | 0 (钢琴) |
| `--sr` | 采样率 | 22050 |

## 项目结构

```
AutoTranscriber/
├── main.py                         # 主入口
├── AutoTranscriber/
│   ├── __init__.py
│   ├── audio_loader.py             # 音频加载与预处理
│   ├── spectral.py                 # 频谱分析 (CQT)
│   ├── onset_detection.py          # 音符起始检测
│   ├── pitch_estimation.py         # 多音高估计（谐波减法）
│   ├── note_tracking.py            # 音符追踪与平滑
│   ├── midi_writer.py              # MIDI 输出
│   └── separator.py                # 音源分离（Demucs + HPSS 后备）
├── test_audio/                     # 测试音频
├── requirements.txt
└── README.md
```

## 依赖

- librosa >= 0.10.0
- numpy, scipy, pretty_midi
- soundfile, av
- Demucs（可选，用于音源分离）

## 算法说明

1. **音源分离**：Demucs (Hybrid Transformer) 将歌曲分离为人声和伴奏
2. **CQT 频谱**：Constant-Q Transform，频率轴对数刻度，符合音乐感知
3. **多音高估计（谐波减法）**：逐帧迭代——检测最强基频后减去其全部谐波成分，再检测下一个频率，从而分离和弦中的各个音符
4. **音符追踪**：将同音高的连续帧连接为完整音符事件，过滤短时噪声
5. **MIDI 输出**：多音轨标准 MIDI 文件，可用任何 DAW 打开

## 示例

```bash
# 和弦音频
python main.py -i test_audio/chord_progression.wav -o chord.mid

# 歌曲扒谱（含人声分离）
python main.py -i 周杰伦.m4a -o 周杰伦.mid --separate
```
