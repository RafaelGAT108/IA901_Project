"""
Module for handling audio data, including loading and visualization.

The features can be obtained using the `transforms` module, which
includes various feature extraction techniques (e.g. spectrograms, MFCCs, etc).
"""

import librosa
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from typing import Any


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


class LungSoundFeatures():
    """
    Representation of extracted features from a lung sound recording.
    """
    def __init__(self, file_path: str | Path | None = None):
        """
        Initialize the lung sound features object.
        Args:
            file_path (str | Path | None): Path to the features file. If None, creates an empty object that can be loaded later.
        """
        self.file_path = Path(file_path) if file_path is not None else None
        self.features: np.ndarray | None = None
        self.sr: int | None = None
        self.info: dict | None = None
        if file_path is not None:
            self.load()

    def load(self):
        """
        Load the features from a file.
        """
        if self.file_path.suffix == ".npy":
            self.features = np.load(self.file_path)
        elif self.file_path.suffix == ".npz":
            data = np.load(self.file_path)
            self.features = data["features"]
            self.sr = int(data["sr"])
        elif self.file_path.suffix in [".png", ".jpg", ".jpeg"]:
            img = Image.open(self.file_path)
            self.features = np.array(img)
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")

    def plot_features(self, title=None, ax=None, **params) -> Any:
        """
        Plot the extracted features (spectrogram representation).
        Args:
            title: Plot title.
            ax: Matplotlib axis to plot on. If None, creates a new figure.
            **params: Additional parameters for the plotting function (e.g. hop_length, x_axis, y_axis, etc).
        Returns:
            The image object created by the plotting function.
        """
        if self.features is None:
            raise ValueError("Features not loaded. Call load() first.")
        if title is None:
            title = f"{self.file_path.stem}"

        # If the features have a single channel dimension, remove it for plotting
        if self.features.ndim == 3 and self.features.shape[-1] == 1:
            features = self.features.squeeze(-1)
        else:
            features = self.features

        # Determine how to plot based on the shape of the features
        if features.ndim == 2:
            sr = self.sr if self.sr is not None else 22050
            img = librosa.display.specshow(features, sr=sr, ax=ax, **params)
            if ax is None:
                plt.title(title)
                plt.colorbar(img)
                plt.show()
            else:
                ax.set_title(title)
        elif features.ndim == 3 and features.shape[-1] == 3:
            if ax is None:
                plt.title(title)
                img = plt.imshow(self.features)
                plt.show()
            else:
                img = ax.imshow(self.features)
                ax.set_title(title)
        else:
            return None  # Unsupported feature shape for plotting