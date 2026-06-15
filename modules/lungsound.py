"""
Module for handling audio data and extracted features, including loading and visualization.

The features can be obtained using the `transforms` module, which
includes various feature extraction techniques (e.g. spectrograms, MFCCs, etc).
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any


class LungSoundAudio():
    """
    Representation of a single lung sound recording.
    """
    def __init__(
            self,
            file_path: str | Path | None = None,
            audio: np.ndarray | None = None,
            sr: int | None = None,
        ) -> None:
        """
        Initialize the lung sound audio object.
        Args:
            file_path (str | Path | None): Path to the audio file. If None, creates an empty object that can be loaded later.
            audio (np.ndarray | None): The audio waveform as a numpy array. If file_path is provided, this will be overwritten by the loaded audio.
            sr (int | None): Sample rate of the audio. If file_path is provided, this will be overwritten by the loaded sample rate.
        """
        self.file_path = Path(file_path)
        self.audio = audio
        self.sr = sr
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
    def __init__(
            self,
            file_path: str | Path | None = None,
            features: np.ndarray | None = None,
            sr: int | None = None,
        ) -> None:
        """
        Initialize the lung sound features object.
        Args:
            file_path (str | Path | None): Path to the features file. If None, creates an empty object that can be loaded later.
            features (np.ndarray | None): The extracted features as a numpy array. If file_path is provided, this will be overwritten by the loaded features.
            sr (int | None): Sample rate associated with the features. If file_path is provided, this will be overwritten by the loaded sample rate.
        """
        self.file_path = Path(file_path) if file_path is not None else None
        self.features = features
        self.sr = sr
        self.info: dict | None = None
        if file_path is not None:
            self.load()

    def load(self) -> tuple[np.ndarray, int]:
        """
        Load the features from a file.
        Assumes the file is a .npz file containing 'features' and 'sr' arrays.
        Can be extended to support other formats in the future.
        """
        if self.file_path.suffix == ".npz":
            data = np.load(self.file_path, allow_pickle=True)
            self.features = data["features"]
            self.sr = int(data["sr"])
        else:
            raise ValueError(f"Unsupported file format: {self.file_path.suffix}")
        return self.features, self.sr

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
        # If it's a 2D array, we assume it's a spectrogram and use librosa's specshow
        if features.ndim == 2 and self.sr is not None:
            img = librosa.display.specshow(features, sr=self.sr, ax=ax, **params)
            if ax is None:
                plt.title(title)
                plt.colorbar(img)
                plt.show()
            else:
                ax.set_title(title)
                plt.colorbar(img, ax=ax)
            return img
        # If it's a 3D array with 3 channels, we assume it's an RGB image and use imshow
        elif features.ndim == 3 and features.shape[-1] == 3:
            if ax is None:
                plt.title(title)
                img = plt.imshow(self.features)
                plt.show()
            else:
                img = ax.imshow(self.features)
                ax.set_title(title)
            return img
        else:
            return None  # Unsupported feature shape for plotting