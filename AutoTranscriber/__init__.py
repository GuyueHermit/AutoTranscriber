"""AutoTranscriber — 音频自动扒谱工具包"""

from .audio_loader import load_audio, preprocess
from .spectral import compute_cqt, compute_spectral_flux, compute_stft
from .onset_detection import detect_onsets
from .pitch_estimation import estimate_pitches, estimate_vocal_pitch, hz_to_midi, midi_to_hz, midi_to_name
from .note_tracking import track_notes
from .midi_writer import write_midi, write_multitrack_midi, merge_tracks_to_piano
from .separator import separate_audio, has_demucs
from .midi_to_pdf import midi_to_pdf, find_musescore, install_musescore_guide
from .crepe_wrapper import estimate_vocal_pitch_crepe, has_crepe

__version__ = "1.0.0"
