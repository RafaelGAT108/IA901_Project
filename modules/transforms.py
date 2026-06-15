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
    >>> lung_sound = LungSoundAudio("sample.wav")
    >>> features = transforms(lung_sound)
    >>> print(features.shape)
"""

import copy
import librosa
import numpy as np
from abc import ABC, abstractmethod
from modules.lungsound import LungSoundAudio, LungSoundFeatures


# ============================================================
# Base classes
# ============================================================


class AudioTransform(ABC):
    """ Base class for 1D -> 1D audio transforms. """
    @abstractmethod
    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundAudio:
        ...

    def __repr__(self):
        params = ",".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
        )
        return f"{self.__class__.__name__}({params})"

    @property
    def name(self):
        return self.__class__.__name__


class FeatureExtractor(ABC):
    """ Base class for 1D -> 2D feature extractors. """
    @abstractmethod
    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        ...

    def __repr__(self):
        params = ",".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
        )
        return f"{self.__class__.__name__}({params})"

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def plot_params(self):
        """ Parameters to use when plotting the features with librosa.display.specshow. """
        return {}

class FeatureTransform(ABC):
    """ Base class for 2D -> 2D feature transforms. """
    @abstractmethod
    def __call__(self, lung_sound_features: LungSoundFeatures) -> LungSoundFeatures:
        ...

    def __repr__(self):
        params = ",".join(
            f"{k}={v!r}"
            for k, v in vars(self).items()
        )
        return f"{self.__class__.__name__}({params})"

    @property
    def name(self):
        return self.__class__.__name__


class Compose(AudioTransform, FeatureExtractor, FeatureTransform):
    """ Compose multiple transforms sequentially. """
    def __init__(self, transforms: list[AudioTransform | FeatureExtractor | FeatureTransform]):
        self.transforms = transforms

    def __call__(self, lung_sound):
        for transform in self.transforms:
            lung_sound = transform(lung_sound)
        return lung_sound


# ============================================================
# Waveform transforms
# 1D -> 1D (audio -> audio)
# 1 sample -> 1 sample OR 1 sample -> N samples
# ============================================================


class NormalizeAudio(AudioTransform):
    """ Normalize waveform amplitude to [-1, 1]. """
    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundAudio:
        audio = lung_sound.audio
        # lung_sound.audio = librosa.util.normalize(audio, axis=None)
        lung_sound.audio = audio / (np.max(np.abs(audio)) + 1e-8)
        return lung_sound


class Resample(AudioTransform):
    """ Resample waveform to a target sample rate. """
    def __init__(self, target_sr: int):
        """
        Args:
            target_sr: Target sample rate in Hz.
        """
        self.target_sr = target_sr

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundAudio:
        audio = lung_sound.audio
        sr = lung_sound.sr
        if sr != self.target_sr:
            lung_sound.audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            lung_sound.sr = self.target_sr
        return lung_sound


class PadOrTrim(AudioTransform):
    """ Pad with zeros or trim waveform to a fixed duration. """
    def __init__(self, target_duration: float):
        """
        Args:
            target_duration: Target duration in seconds.
        """
        self.target_duration = target_duration

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundAudio:
        audio = lung_sound.audio
        sr = lung_sound.sr
        target_len = int(self.target_duration * sr)
        if len(audio) < target_len:
            lung_sound.audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
        else:
            lung_sound.audio = audio[:target_len]
        return lung_sound


class Crop(AudioTransform):
    """ Crop a random segment of the audio to a fixed duration. """
    def __init__(self, start_time: float, end_time: float):
        """
        Args:
            start_time: Start time of the crop in seconds.
            end_time: End time of the crop in seconds.
        """
        self.start_time = start_time
        self.end_time = end_time

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundAudio:
        audio = lung_sound.audio
        sr = lung_sound.sr
        start_sample = int(self.start_time * sr)
        end_sample = int(self.end_time * sr)
        lung_sound.audio = audio[start_sample:end_sample]
        return lung_sound


class Window(AudioTransform):
    """
    Split the audio into N windows of fixed duration.
    1 sample -> N samples (e.g. one long audio -> multiple 5-second crops)
    """
    def __init__(self, window_length: float, hop_length: float):
        """
        Initializes the Window transform.
        Args:
            window_length: Length of the window in seconds.
            hop_length: Hop length between windows in seconds.
        """
        self.window_length = window_length
        self.hop_length = hop_length

    def __call__(self, lung_sound: LungSoundAudio) -> list[LungSoundAudio]:
        duration = lung_sound.duration
        start = 0.0
        end = start + self.window_length
        crops = []
        while end <= duration:
            cropped = copy.copy(lung_sound)
            cropped = Crop(start_time=start, end_time=end)(cropped)
            crops.append(cropped)
            start += self.hop_length
            end += self.hop_length
        return crops


# ============================================================
# Feature extractors
# 1D -> N-D (audio -> features)
# ============================================================


class STFT(FeatureExtractor):
    """ Standard STFT spectrogram (complex-valued). """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        """
        Args:
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        stft = librosa.stft(
            lung_sound.audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features = stft
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class MagSTFT(STFT):
    """ Standard magnitude spectrogram in dB scale. """
    @property
    def plot_params(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "y_axis": "log",
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        stft = super().__call__(lung_sound).features
        mag = np.abs(stft)
        features = librosa.amplitude_to_db(mag, ref=np.max)
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class ImagSTFT(STFT):
    """ Imaginary part of the STFT spectrogram. """
    @property
    def plot_params(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "y_axis": "log",
            "x_axis": "time",
        }
    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        stft = super().__call__(lung_sound).features
        features = stft.imag
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class RealSTFT(STFT):
    """ Real part of the STFT spectrogram. """
    @property
    def plot_params(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "y_axis": "log",
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        stft = super().__call__(lung_sound).features
        features = stft.real
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class MelSpectrogram(FeatureExtractor):
    """ Mel spectrogram in dB scale. """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
        """
        Args:
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
            n_mels: Number of Mel bands to generate.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    @property
    def plot_params(self):
        return {
            "hop_length": self.hop_length,
            "y_axis": "mel",
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        mel_spec = librosa.feature.melspectrogram(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        features = librosa.power_to_db(mel_spec, ref=np.max)
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class MFCC(FeatureExtractor):
    """ Mel-frequency cepstral coefficients. """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, n_mfcc: int = 20):
        """
        Args:
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
            n_mfcc: Number of MFCC coefficients to compute.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc

    @property
    def plot_params(self):
        return {
            "hop_length": self.hop_length,
            "y_axis": None,
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        features = librosa.feature.mfcc(
            y=lung_sound.audio,
            sr=lung_sound.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mfcc=self.n_mfcc
        )
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class MFCCDelta(MFCC):
    """ Temporal derivative of MFCC coefficients. """
    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        mfcc_features = super().__call__(lung_sound).features
        features = librosa.feature.delta(mfcc_features)
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class Chroma(FeatureExtractor):
    """ Chroma spectrogram. """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, n_chroma: int = 12):
        """
        Args:
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
            n_chroma: Number of chroma bins to generate.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    @property
    def plot_params(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "y_axis": "chroma",
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        stft = librosa.stft(lung_sound.audio)
        mag = np.abs(stft)
        features = librosa.feature.chroma_stft(
            S=mag,
            sr=lung_sound.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma
        )
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class SpectralContrast(FeatureExtractor):
    """ Spectral contrast (amplitude peaks vs valleys in frequency bands). """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512, n_bands: int = 4, fmin: float = 50):
        """
        Args:
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
            n_bands: Number of frequency bands to use for contrast calculation.
            fmin: Minimum frequency to consider in Hz.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bands = n_bands
        self.fmin = fmin

    @property
    def plot_params(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "y_axis": None,
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        stft = librosa.stft(
            lung_sound.audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        mag = np.abs(stft)
        features = librosa.feature.spectral_contrast(
            S=mag,
            sr=lung_sound.sr,
            n_bands=self.n_bands,
            fmin=self.fmin
        )
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class CQT(FeatureExtractor):
    """ Constant-Q transform spectrogram (logarithmically spaced frequency bins). """
    def __init__(self, n_bins: int = 48, fmin: float = 30, bins_per_octave: int = 12):
        """
        Args:
            n_bins: Number of frequency bins to use.
            fmin: Minimum frequency to consider in Hz.
            bins_per_octave: Number of bins per octave.
        """
        self.n_bins = n_bins
        self.fmin = fmin
        self.bins_per_octave = bins_per_octave

    @property
    def plot_params(self):
        return {
            "fmin": self.fmin,
            "bins_per_octave": self.bins_per_octave,
            "y_axis": "cqt_note",
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        cqt = librosa.cqt(
            lung_sound.audio,
            sr=lung_sound.sr,
            n_bins=self.n_bins,
            fmin=self.fmin,
            bins_per_octave=self.bins_per_octave
        )
        mag = np.abs(cqt)
        features = librosa.amplitude_to_db(mag, ref=np.max)
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class Phase(FeatureExtractor):
    """ Phase spectrogram (angle of STFT coefficients). """
    def __init__(self, n_fft: int = 2048, hop_length: int = 512):
        """
        Args:
            n_fft: Length of the FFT window in samples.
            hop_length: Number of samples between successive frames.
        """
        self.n_fft = n_fft
        self.hop_length = hop_length

    @property
    def plot_params(self):
        return {
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "y_axis": "log",
            "x_axis": "time",
        }

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        stft = librosa.stft(
            lung_sound.audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        features = np.angle(stft)
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class SinPhase(Phase):
    """ Sine of the phase spectrogram to capture periodicity. """
    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        phase_features = super().__call__(lung_sound).features
        features = np.sin(phase_features)
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


class CosPhase(Phase):
    """ Cosine of the phase spectrogram to capture periodicity. """
    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundFeatures:
        phase_features = super().__call__(lung_sound).features
        features = np.cos(phase_features)
        return LungSoundFeatures(features=features, sr=lung_sound.sr)


# ============================================================
# Feature transforms
# N-D -> N-D (features -> features)
# ============================================================


class MinMaxNormalization(FeatureTransform):
    """ Normalize features to mantain the values between 0 and 1. """
    def __init__(self, mode: str = "channel_wise"):
        """
        Args:
            mode (str): "global" to compute min/max across all values, "channel_wise" to compute separately for each channel.
        """
        assert mode in ["global", "channel_wise"], "mode must be 'global' or 'channel_wise'"
        self.mode = mode

    def __call__(self, lung_sound_features: LungSoundFeatures) -> LungSoundFeatures:
        features = lung_sound_features.features
        if self.mode == "global":
            min_val = np.min(features)
            max_val = np.max(features)
        else:
            min_val = np.min(features, axis=(0, 1), keepdims=True)
            max_val = np.max(features, axis=(0, 1), keepdims=True)
        normalized = (features - min_val) / (max_val - min_val + 1e-8)
        lung_sound_features.features = normalized
        return lung_sound_features


class ZScoreNormalization(FeatureTransform):
    """ Standardize features to have zero mean and unit variance. """
    def __init__(self, mode: str = "channel_wise"):
        """
        Args:
            mode (str): "global" to compute mean/std across all values, "channel_wise" to compute separately for each channel.
        """
        assert mode in ["global", "channel_wise"], "mode must be 'global' or 'channel_wise'"
        self.mode = mode

    def __call__(self, lung_sound_features: LungSoundFeatures) -> LungSoundFeatures:
        features = lung_sound_features.features
        if self.mode == "global":
            mean = np.mean(features)
            std = np.std(features)
        else:
            mean = np.mean(features, axis=(0, 1), keepdims=True)
            std = np.std(features, axis=(0, 1), keepdims=True)
        standardized = (features - mean) / (std + 1e-8)
        lung_sound_features.features = standardized
        return lung_sound_features


class ImageNetNormalization(FeatureTransform):
    """ Normalize features using ImageNet mean and std for 3-channel inputs. """
    def __call__(self, lung_sound_features: LungSoundFeatures) -> LungSoundFeatures:
        features = lung_sound_features.features
        assert features.ndim == 3 and features.shape[2] == 3, "ImageNetNormalization expects 3-channel features"
        assert np.all((features >= 0) & (features <= 1)), "ImageNetNormalization expects features in [0, 1] range"
        mean = np.array([0.485, 0.456, 0.406])[np.newaxis, np.newaxis, :]
        std = np.array([0.229, 0.224, 0.225])[np.newaxis, np.newaxis, :]
        normalized = (features - mean) / std
        lung_sound_features.features = normalized
        return lung_sound_features