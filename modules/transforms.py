"""
Audio processing for lung sound datasets.

This module provides:
- Waveform transforms (normalization, resampling, padding, etc.)
- Compose for chaining transforms

Examples:
    >>> transforms = Compose([
    ...     Resample(20500),
    ...     PadOrTrim(5),
    ...     Normalize(),
    ... ])
    >>>
    >>> lung_sound = LungSoundAudio("sample.wav")
    >>> transforms(lung_sound)
    >>> print(lung_sound.audio.shape)
"""

import copy
import librosa
import numpy as np
from abc import ABC, abstractmethod
from modules.lungsound import LungSoundAudio

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
        self.transforms = transforms

    def __call__(self, lung_sound: LungSoundAudio) -> LungSoundAudio:
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