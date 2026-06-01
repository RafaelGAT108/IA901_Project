"""
Module for handling audio data, including loading and visualization.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class LungSoundAudio():
    """
    Representation of a single lung sound recording.
    """
    def __init__(self, file_path: str | Path | None = None):
        """
        Initialize the lung sound audio object.
        Args:
            file_path (str | Path | None): Path to the audio file. If None, creates an empty object that can be loaded later.
        """
        self.file_path = Path(file_path)
        self.audio: np.ndarray | None = None
        self.sr: int | None = None
        self.info: dict | None = None
        if file_path is not None:
            self.load()

    @property
    def duration(self) -> float | None:
        if self.audio is None or self.sr is None:
            return None
        return len(self.audio) / self.sr

    def load(self, sr: int | None = None, mono: bool = True) -> tuple[np.ndarray, int]:
        """
        Load the audio file into memory.
        Args:
            sr: Target sample rate. If None, uses the original sample rate.
            mono: Whether to convert to mono by averaging channels.
        Returns:
            A tuple of (audio waveform, sample rate).
        """
        self.audio, self.sr = librosa.load(self.file_path, sr=sr, mono=mono)
        return self.audio, self.sr

    def plot_waveform(self, title=None, ax=None) -> librosa.display.AdaptiveWaveplot:
        """
        Plot the audio waveform.
        Args:
            title: Plot title.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
        """
        if self.audio is None or self.sr is None:
            raise ValueError("Audio data not loaded. Call load() first.")
        if title is None:
            title = self.file_path.name
        img = librosa.display.waveshow(self.audio, sr=self.sr, ax=ax)
        if ax is None:
            plt.title(title)
            plt.show()
        else:
            ax.set_title(title)
        return img