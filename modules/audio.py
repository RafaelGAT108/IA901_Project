"""
Module for handling audio data, including loading and visualization.

The features can be obtained using the `transforms` module, which
includes various feature extraction techniques (e.g. spectrograms, MFCCs, etc).
"""

from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh


class LungSound():
    """
    Representation of a single lung sound recording.
    """
    def __init__(self, file_path):
        self.wav_file = Path(file_path)
        self.audio: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.feature_extractor: str | None = None
        self.sr: int | None = None
        self.load()
        # NOTE: In case we want to apply windowing, we can store the start and end
        # times of the current window in respect to the original audio file
        # This is handled by the `preprocess` module
        self.window_start = None
        self.window_end = None

    @property
    def duration(self) -> float | None:
        if self.audio is None or self.sr is None:
            return None
        return len(self.audio) / self.sr

    @property
    def has_features(self):
        return self.features is not None

    def load(self, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
        """
        Load the audio file into memory.
        Args:
            sr: Target sample rate. If None, uses the original sample rate.
            mono: Whether to convert to mono by averaging channels.
        Returns:
            A tuple of (audio waveform, sample rate).
        """
        self.audio, self.sr = librosa.load(self.wav_file, sr=sr, mono=mono)
        return self.audio, self.sr

    def plot_waveform(self, title=None, ax=None):
        """
        Plot the audio waveform.
        Args:
            title: Plot title.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
        """
        if title is None:
            title = self.wav_file.name
        librosa.display.waveshow(self.audio, sr=self.sr, ax=ax)
        if ax is None:
            plt.title(title)
            plt.show()
        else:
            ax.set_title(title)

    def plot_features(self, title=None, ax=None, **params) -> QuadMesh:
        """
        Plot the extracted features (spectrogram representation).
        Args:
            title: Plot title.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            **params: Additional parameters to pass to librosa.display.specshow.
        Returns:
            The QuadMesh object created by librosa.display.specshow.
        """
        if not self.has_features:
            raise ValueError("No features to plot. Please extract features first.")
        if title is None:
            title = self.wav_file.name
        img = librosa.display.specshow(self.features, sr=self.sr, ax=ax, **params)
        if ax is None:
            plt.title(title)
            plt.show()
        else:
            ax.set_title(title)
        return img