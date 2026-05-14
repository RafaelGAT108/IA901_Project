"""
Audio feature extraction transforms for lung sound datasets.

Each transform extracts a single type of audio representation from a LungSound object.
Use one transform per dataset, depending on your use case.

Examples:
    >>> from dataset import ICBHIDataset, MelSpectrogramTransform
    >>> transform = MelSpectrogramTransform(n_mels=128)
    >>> dataset = ICBHIDataset(root_path, transform=transform)
    >>> feature, label = dataset[0]
    >>> print(feature.shape)  # (128, time_steps)
"""

import librosa
import numpy as np
from abc import ABC, abstractmethod
from dataset.audio import LungSound


class AudioTransform(ABC):
    """
    Base class for audio feature extraction transforms.
    Each transform extracts a specific audio representation (spectrogram, MFCC, etc).
    """
    @abstractmethod
    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        """Extract a feature representation from audio.
        Args:
            lung_sound: A LungSound instance with loaded audio.
        Returns:
            A numpy array representing the extracted feature.
        """
        pass


class SpectrogramTransform(AudioTransform):
    """
    Standard spectrogram (magnitude of STFT in dB scale).
    Output shape: (freq_bins, time_steps) = (2048, time_steps)
    """
    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        stft = librosa.stft(lung_sound.audio)
        mag = np.abs(stft)
        return librosa.amplitude_to_db(mag, ref=np.max)


class MelSpectrogramTransform(AudioTransform):
    """
    Mel spectrogram (perceptually-motivated frequency scale).
    Output shape: (n_mels, time_steps) | default (128, time_steps)
    Args:
        n_mels: Number of Mel bands (default 128).
    """
    def __init__(self, n_mels: int = 128):
        self.n_mels = n_mels

    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_mels=self.n_mels,
        )
        return librosa.power_to_db(mel_spec, ref=np.max)


class MFCCTransform(AudioTransform):
    """
    Mel-Frequency Cepstral Coefficients (compact spectral summary).
    Output shape: (n_mfcc, time_steps) | default (13, time_steps)
    Args:
        n_mfcc: Number of MFCC coefficients (default 13).
    """
    def __init__(self, n_mfcc: int = 20):
        self.n_mfcc = n_mfcc

    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        return librosa.feature.mfcc(y=lung_sound.audio, sr=lung_sound.sr, n_mfcc=self.n_mfcc)


class MFCCDeltaTransform(AudioTransform):
    """
    Delta of MFCCs (temporal derivative of MFCC features).
    Output shape: (n_mfcc, time_steps) | default (20, time_steps)
    Args:
        n_mfcc: Number of MFCC coefficients to compute delta for (default 20).
    """
    def __init__(self, n_mfcc: int = 20):
        self.n_mfcc = n_mfcc

    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        mfcc = librosa.feature.mfcc(y=lung_sound.audio, sr=lung_sound.sr, n_mfcc=self.n_mfcc)
        return librosa.feature.delta(mfcc)


class ChromaTransform(AudioTransform):
    """
    Chroma features (pitch-class distribution, 12 chromatic notes).
    Output shape: (12, time_steps)
    """
    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        stft = librosa.stft(lung_sound.audio)
        mag = np.abs(stft)
        return librosa.feature.chroma_stft(S=mag, sr=lung_sound.sr)


class SpectralContrastTransform(AudioTransform):
    """Spectral contrast (amplitude peaks vs valleys in frequency bands).
    Output shape: (n_bands, time_steps) | default (6, time_steps)
    Args:
        n_bands: Number of frequency bands (default 6).
    """
    def __init__(self, n_bands: int = 4, fmin: float = 50):
        self.n_bands = n_bands
        self.fmin = fmin

    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        stft = librosa.stft(lung_sound.audio)
        mag = np.abs(stft)
        return librosa.feature.spectral_contrast(
            S=mag, 
            sr=lung_sound.sr,
            fmin=self.fmin,
            n_bands=self.n_bands
        )


class CQTSpectrogramTransform(AudioTransform):
    """Constant-Q Transform spectrogram (logarithmically spaced frequency bins).
    Output shape: (n_bins, time_steps)
    Args:
        n_bins: Number of frequency bins (default 48).
        fmin: Minimum frequency (default 30).
        bins_per_octave: Number of bins per octave (default 12).
    """
    def __init__(self, n_bins: int = 48, fmin: float = 30, bins_per_octave: int = 12):
        self.n_bins = n_bins
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave

    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        cqt = librosa.cqt(
            lung_sound.audio,
            sr=lung_sound.sr,
            n_bins=self.n_bins,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave
        )
        return librosa.amplitude_to_db(np.abs(cqt), ref=np.max)


class PhaseTransform(AudioTransform):
    """
    Phase spectrogram (angle of STFT coefficients).
    Output shape: (freq_bins, time_steps) = (2048, time_steps)
    """
    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        stft = librosa.stft(lung_sound.audio)
        return np.angle(stft)


class RawWaveformTransform(AudioTransform):
    """Standardized raw audio waveform (resampled, padded, normalized).
    Output shape: (target_sr x target_duration,)
    Args:
        target_sr: Target sample rate (default 16000).
        target_duration: Target duration in seconds (default 20).
    """
    def __init__(self, target_sr: int = 16000, target_duration: int = 20):
        self.target_sr = target_sr
        self.target_duration = target_duration

    def __call__(self, lung_sound: LungSound) -> np.ndarray:
        audio, _ = lung_sound.standardize(self.target_sr, self.target_duration)
        return audio