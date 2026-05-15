"""
Audio preprocessing and feature extraction for lung sound datasets.

This module provides:
- Waveform transforms (normalization, resampling, padding, etc.)
- Feature extractors (spectrograms, MFCCs, chroma, etc.)
- Compose for chaining transforms

Examples:
    >>> preprocess = Compose([
    ...     Resample(16000),
    ...     PadOrTrim(5),
    ...     Normalize(),
    ...     MelSpectrogram(n_mels=128),
    ... ])
    >>>
    >>> lung_sound = LungSound("sample.wav")
    >>> lung_sound = preprocess(lung_sound)
    >>> print(lung_sound.features.shape)
"""

import librosa
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from modules.audio import LungSound


# ============================================================
# Base classes
# ============================================================


class AudioTransform(ABC):
    """
    Base class for all audio transforms.
    """
    @abstractmethod
    def __call__(self, lung_sound: LungSound) -> LungSound:
        pass


class FeatureExtractor(AudioTransform):
    """
    Base class for transforms that extract features from waveform audio.
    """
    pass


class Compose(AudioTransform):
    """
    Compose multiple transforms sequentially.
    Example:
    >>> Compose([
    ...     Resample(16000),
    ...     PadOrTrim(5),
    ...     Normalize(),
    ...     MelSpectrogram(),
    ... ])
    """
    def __init__(self, transforms: list[AudioTransform]):
        self.name = "Compose"
        self.transforms = transforms

    def __call__(self, lung_sound: LungSound) -> LungSound:
        for transform in self.transforms:
            lung_sound = transform(lung_sound)
        return lung_sound


# ============================================================
# Waveform transforms
# ============================================================


class Normalize(AudioTransform):
    """
    Normalize waveform amplitude to [-1, 1].
    """
    def __init__(self):
        self.name = "Normalize"

    def __call__(self, lung_sound: LungSound) -> LungSound:
        audio = lung_sound.audio
        # lung_sound.audio = librosa.util.normalize(audio, axis=None)
        lung_sound.audio = audio / (np.max(np.abs(audio)) + 1e-8)
        lung_sound.features = None  # Clear features since the original audio has changed
        return lung_sound


class Resample(AudioTransform):
    """
    Resample waveform to a target sample rate.
    """
    def __init__(self, target_sr: int):
        self.name = "Resample"
        self.target_sr = target_sr

    def __call__(self, lung_sound: LungSound) -> LungSound:
        audio = lung_sound.audio
        sr = lung_sound.sr
        if sr != self.target_sr:
            lung_sound.audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            lung_sound.sr = self.target_sr
        lung_sound.features = None  # Clear features since the original audio has changed
        return lung_sound


class PadOrTrim(AudioTransform):
    """
    Pad with zeros or trim waveform to a fixed duration.
    Args:
        target_duration:
            Target duration in seconds.
    """
    def __init__(self, target_duration: float):
        self.name = "PadOrTrim"
        self.target_duration = target_duration

    def __call__(self, lung_sound: LungSound) -> LungSound:
        audio = lung_sound.audio
        sr = lung_sound.sr
        target_len = int(self.target_duration * sr)
        if len(audio) < target_len:
            lung_sound.audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        else:
            lung_sound.audio = audio[:target_len]
        lung_sound.features = None  # Clear features since the original audio has changed
        return lung_sound


class Window(AudioTransform):
    """
    Extract a fixed-length window from the audio signal.
    Args:
        window_length: Length of the window in seconds.
        hop_length: Hop length between windows in seconds. If None, defaults to window_length (non-overlapping windows).
    """
    def __init__(self, window_length: float, hop_length: float | None = None):
        self.name = "Window"
        self.window_length = window_length
        self.hop_length = hop_length or window_length

    def expand_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            duration = librosa.get_duration(path=row["FilePath"])
            start = 0.0
            end = start + self.window_length
            while end <= duration:
                new_row = row.to_dict()
                new_row["OriginalDuration"] = duration
                new_row["Start"] = start
                new_row["End"] = end
                records.append(new_row)
                start += self.hop_length
                end += self.hop_length
        return pd.DataFrame(records)

    def __call__(self, lung_sound: LungSound) -> LungSound:
        # Check if windowing parameters are set
        start = lung_sound.window_start
        end = lung_sound.window_end
        if start is None or end is None:
            return lung_sound  # No windowing applied
        # Convert start and end times to sample indices
        audio = lung_sound.audio
        sr = lung_sound.sr
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        lung_sound.audio = (audio[start_sample:end_sample])
        lung_sound.features = None  # Clear features since the original audio has changed
        return lung_sound

# ============================================================
# Feature extractors
# ============================================================


class Spectrogram(FeatureExtractor):
    """
    Standard magnitude spectrogram in dB scale.
    """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        self.name = "Spectrogram"
        self.y_axis = "log"
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, lung_sound: LungSound) -> LungSound:
        stft = librosa.stft(
            lung_sound.audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mag = np.abs(stft)
        lung_sound.features = librosa.amplitude_to_db(mag, ref=np.max)
        lung_sound.feature_extractor = self.name
        return lung_sound


class MelSpectrogram(FeatureExtractor):
    """
    Mel spectrogram in dB scale.
    """
    def __init__(self, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512):
        self.name = "MelSpectrogram"
        self.y_axis = "mel"
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, lung_sound: LungSound) -> LungSound:
        mel_spec = librosa.feature.melspectrogram(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        lung_sound.features = librosa.power_to_db(mel_spec, ref=np.max)
        lung_sound.feature_extractor = self.name
        return lung_sound


class MFCC(FeatureExtractor):
    """
    Mel-frequency cepstral coefficients.
    """
    def __init__(self, n_mfcc: int = 20):
        self.name = "MFCC"
        self.y_axis = None
        self.n_mfcc = n_mfcc

    def __call__(self, lung_sound: LungSound) -> LungSound:
        lung_sound.features = librosa.feature.mfcc(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_mfcc=self.n_mfcc
        )
        lung_sound.feature_extractor = self.name
        return lung_sound


class MFCCDelta(FeatureExtractor):
    """
    Temporal derivative of MFCC coefficients.
    """
    def __init__(self, n_mfcc: int = 20):
        self.name = "MFCCDelta"
        self.y_axis = None
        self.n_mfcc = n_mfcc

    def __call__(self, lung_sound: LungSound) -> LungSound:
        mfcc = librosa.feature.mfcc(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_mfcc=self.n_mfcc
        )
        lung_sound.features = librosa.feature.delta(mfcc)
        lung_sound.feature_extractor = self.name
        return lung_sound


class ChromaSpectrogram(FeatureExtractor):
    """
    Chroma spectrogram.
    """
    def __init__(self):
        self.name = "ChromaSpectrogram"
        self.y_axis = "chroma"

    def __call__(self, lung_sound: LungSound) -> LungSound:
        stft = librosa.stft(lung_sound.audio)
        mag = np.abs(stft)
        lung_sound.features = librosa.feature.chroma_stft(S=mag, sr=lung_sound.sr)
        lung_sound.feature_extractor = self.name
        return lung_sound


class SpectralContrast(FeatureExtractor):
    """
    Spectral contrast (amplitude peaks vs valleys in frequency bands).
    """
    def __init__(self, n_bands: int = 4, fmin: float = 50):
        self.name = "SpectralContrast"
        self.y_axis = None
        self.n_bands = n_bands
        self.fmin = fmin

    def __call__(self, lung_sound: LungSound) -> LungSound:
        stft = librosa.stft(lung_sound.audio)
        mag = np.abs(stft)
        lung_sound.features = librosa.feature.spectral_contrast(
            S=mag,
            sr=lung_sound.sr,
            n_bands=self.n_bands,
            fmin=self.fmin
        )
        lung_sound.feature_extractor = self.name
        return lung_sound


class CQTSpectrogram(FeatureExtractor):
    """
    Constant-Q transform spectrogram (logarithmically spaced frequency bins).
    """
    def __init__(self, n_bins: int = 48, fmin: float = 30, bins_per_octave: int = 12):
        self.name = "CQTSpectrogram"
        self.y_axis = "cqt_note"
        self.n_bins = n_bins
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave

    def __call__(self, lung_sound: LungSound) -> LungSound:
        cqt = librosa.cqt(
            lung_sound.audio,
            sr=lung_sound.sr,
            n_bins=self.n_bins,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave
        )
        mag = np.abs(cqt)
        lung_sound.features = librosa.amplitude_to_db(mag, ref=np.max)
        lung_sound.feature_extractor = self.name
        return lung_sound


class Phase(FeatureExtractor):
    """
    Phase spectrogram (angle of STFT coefficients).
    """
    def __init__(self):
        self.name = "Phase"
        self.y_axis = None

    def __call__(self, lung_sound: LungSound) -> LungSound:
        stft = librosa.stft(lung_sound.audio)
        lung_sound.features = np.angle(stft)
        lung_sound.feature_extractor = self.name
        return lung_sound