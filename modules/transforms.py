"""
Audio processing and feature extraction for lung sound datasets.

This module provides:
- Waveform transforms (normalization, resampling, padding, etc.)
- Feature extractors (spectrograms, MFCCs, chroma, etc.)
- Compose for chaining transforms

Examples:
    >>> transforms = Compose([
    ...     Resample(16000),
    ...     PadOrTrim(5),
    ...     Normalize(),
    ...     MelSpectrogram(n_mels=128),
    ... ])
    >>>
    >>> lung_sound = LungSound("sample.wav")
    >>> lung_sound = transforms(lung_sound)
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
    """ Base class for 1D -> 1D audio transforms. """
    @abstractmethod
    def __call__(self, lung_sound: LungSound) -> LungSound:
        pass


class FeatureExtractor(ABC):
    """ Base class for 1D -> 2D feature extractors. """
    @abstractmethod
    def __call__(self, lung_sound: LungSound) -> LungSound:
        pass


class DatasetTransform(ABC):
    """ Base class for transforms that operate on the full dataset. """
    @abstractmethod
    def modify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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
    def __init__(self, transforms: list[AudioTransform | FeatureExtractor | DatasetTransform]):
        self.name = "Compose"
        self.transforms = transforms

    def __call__(self, lung_sound: LungSound) -> LungSound:
        for transform in self.transforms:
            lung_sound = transform(lung_sound)
        return lung_sound


# ============================================================
# Waveform transforms
# 1D -> 1D (audio -> audio)
# ============================================================


class NormalizeAudio(AudioTransform):
    """
    Normalize waveform amplitude to [-1, 1].
    """
    def __init__(self):
        """ Initializes the Normalize transform. """
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
        """
        Initializes the Resample transform.
        Args:
            target_sr: Target sample rate in Hz.
        """
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
    """
    def __init__(self, target_duration: float):
        """
        Initializes the PadOrTrim transform.
        Args:
            target_duration: Target duration in seconds.
        """
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


class Crop(AudioTransform):
    """
    Crop a random segment of the audio to a fixed duration.
    """
    def __init__(self, start_time: float, end_time: float):
        """
        Initializes the Crop transform.
        Args:
            start_time: Start time of the crop in seconds.
            end_time: End time of the crop in seconds.
        """
        self.name = "Crop"
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, lung_sound: LungSound) -> LungSound:
        audio = lung_sound.audio
        sr = lung_sound.sr
        start_sample = int(self.start_time * sr)
        end_sample = int(self.end_time * sr)
        lung_sound.audio = audio[start_sample:end_sample]
        lung_sound.features = None  # Clear features since the original audio has changed
        return lung_sound


# ============================================================
# Dataset-level transforms
# Apply transformations not only to the audio but also to
# the entire dataset structure.
# ============================================================

class Window(DatasetTransform):
    """
    Split audio into fixed-length windows.
    This is a dataset-level transform that expands the dataframe with new rows for each window.
    """
    def __init__(self, window_length: float, hop_length: float | None = None):
        """
        Initializes the Window transform.
        Args:
            window_length: Length of each window in seconds.
            hop_length: Hop length between windows in seconds. If None, defaults to window_length (non-overlapping windows).
        """
        self.name = "Window"
        self.window_length = window_length
        self.hop_length = hop_length or window_length

    def modify_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            duration = row["AudioDuration"]
            start = 0.0
            end = start + self.window_length
            while end <= duration:
                new_row = row.to_dict()
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
        return Crop(start_time=start, end_time=end)(lung_sound)


# ============================================================
# Feature extractors
# 1D -> 2D (audio -> features)
# ============================================================


class Spectrogram(FeatureExtractor):
    """
    Standard magnitude spectrogram in dB scale.
    """
    def __init__(self, n_fft: int = 2048, hop_length: int | None = None):
        """
        Initializes the Spectrogram feature extractor.
        Args:
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
        """
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
        lung_sound.feature_extractor = self
        return lung_sound


class MelSpectrogram(FeatureExtractor):
    """
    Mel spectrogram in dB scale.
    """
    def __init__(self, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512):
        """
        Initializes the MelSpectrogram feature extractor.
        Args:
            n_mels: Number of Mel bands to generate.
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
        """
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
        lung_sound.feature_extractor = self
        return lung_sound


class MFCC(FeatureExtractor):
    """
    Mel-frequency cepstral coefficients.
    """
    def __init__(self, n_mfcc: int = 20):
        """
        Initializes the MFCC feature extractor.
        Args:
            n_mfcc: Number of MFCC coefficients to compute.
        """
        self.n_mfcc = n_mfcc

    def __call__(self, lung_sound: LungSound) -> LungSound:
        lung_sound.features = librosa.feature.mfcc(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_mfcc=self.n_mfcc
        )
        lung_sound.feature_extractor = self
        return lung_sound


class MFCCDelta(FeatureExtractor):
    """
    Temporal derivative of MFCC coefficients.
    """
    def __init__(self, n_mfcc: int = 20):
        """
        Initializes the MFCCDelta feature extractor.
        Args:
            n_mfcc: Number of MFCC coefficients to compute before taking the delta.
        """
        self.n_mfcc = n_mfcc

    def __call__(self, lung_sound: LungSound) -> LungSound:
        mfcc = librosa.feature.mfcc(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_mfcc=self.n_mfcc
        )
        lung_sound.features = librosa.feature.delta(mfcc)
        lung_sound.feature_extractor = self
        return lung_sound


class Chroma(FeatureExtractor):
    """
    Chroma spectrogram.
    """
    def __call__(self, lung_sound: LungSound) -> LungSound:
        stft = librosa.stft(lung_sound.audio)
        mag = np.abs(stft)
        lung_sound.features = librosa.feature.chroma_stft(S=mag, sr=lung_sound.sr)
        lung_sound.feature_extractor = self
        return lung_sound


class SpectralContrast(FeatureExtractor):
    """
    Spectral contrast (amplitude peaks vs valleys in frequency bands).
    """
    def __init__(self, n_bands: int = 4, fmin: float = 50):
        """
        Initializes the SpectralContrast feature extractor.
        Args:
            n_bands: Number of frequency bands to use for contrast calculation.
            fmin: Minimum frequency to consider in Hz.
        """
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
        lung_sound.feature_extractor = self
        return lung_sound


class CQT(FeatureExtractor):
    """
    Constant-Q transform spectrogram (logarithmically spaced frequency bins).
    """
    def __init__(self, n_bins: int = 48, fmin: float = 30, bins_per_octave: int = 12):
        """
        Initializes the CQT feature extractor.
        Args:
            n_bins: Number of frequency bins to use.
            fmin: Minimum frequency to consider in Hz.
            bins_per_octave: Number of bins per octave.
        """
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
        lung_sound.feature_extractor = self
        return lung_sound


class Phase(FeatureExtractor):
    """
    Phase spectrogram (angle of STFT coefficients).
    """
    def __call__(self, lung_sound: LungSound) -> LungSound:
        stft = librosa.stft(lung_sound.audio)
        lung_sound.features = np.angle(stft)
        lung_sound.feature_extractor = self
        return lung_sound