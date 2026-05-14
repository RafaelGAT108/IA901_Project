from __future__ import annotations

import os
import re
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class LungSound:
    """
    Representation of a single lung sound recording.
    """
    def __init__(self, file_path):
        self.wav_file = Path(file_path)
        self.audio = None
        self.sr = None
        self.load()

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

    @staticmethod
    def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resample a waveform to a new sample rate.
        Args:
            audio: The audio waveform.
            orig_sr: The original sample rate.
            target_sr: The target sample rate.
        Returns:
            The resampled audio waveform.
        """
        if orig_sr == target_sr:
            return audio
        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def pad_or_trim(audio: np.ndarray, target_len: int) -> np.ndarray:
        """
        Pad with zeros or trim to a fixed length.
        Args:
            audio: The audio waveform.
            target_len: The target length.
        Returns:
            The padded or trimmed audio waveform.
        """
        if len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)))
        return audio[:target_len]

    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """
        Normalize waveform amplitude to the [-1, 1] range.
        Args:
            audio: The audio waveform.
        Returns:
            The normalized audio waveform.
        """
        return audio / (np.max(np.abs(audio)) + 1e-8)

    def standardize(self, target_sr: int, target_duration: int) -> tuple[np.ndarray, int]:
        """
        Return a resampled, padded and normalized copy of the waveform.
        Args:
            target_sr: The target sample rate.
            target_duration: The target duration in seconds.
        Returns:
            A tuple of (standardized audio waveform, sample rate).
        """
        # print(f"[DEBUG] Before standardization: sr={self.sr}, duration={self.duration:.2f}s, audio shape={self.audio.shape}")
        audio = self.audio
        sr = self.sr

        if sr != target_sr:
            self.audio = self.resample_audio(audio, sr, target_sr)
            self.sr = target_sr

        target_len = int(target_sr * target_duration)
        self.audio = self.pad_or_trim(self.audio, target_len)
        self.audio = self.normalize(self.audio)
        # print(f"[DEBUG] After standardization: sr={self.sr}, duration={self.duration:.2f}s, audio shape={self.audio.shape}")
        return self.audio, self.sr

    def apply_transform(self, transform) -> np.ndarray:
        """
        Apply a given audio transform to this lung sound.
        Args:
            transform: An instance of AudioTransform to apply.
        Returns:
            The transformed audio features as a numpy array.
        """
        self.audio = transform(self)

    @property
    def duration(self) -> float | None:
        if self.audio is None or self.sr is None:
            return None
        return len(self.audio) / self.sr

    def plot_audio(self, title=None, y_axis=None, ax=None):
        """Plot waveform (1D) or spectrogram (2D)."""
        if title is None:
            title = self.wav_file.name
        if self.audio.ndim == 1:
            librosa.display.waveshow(self.audio, sr=self.sr, ax=ax)
        elif self.audio.ndim == 2:
            librosa.display.specshow(self.audio, sr=self.sr, x_axis='time', y_axis=y_axis, ax=ax)
        else:
            raise ValueError("Audio must be 1D or 2D for plotting.")
        if ax is None:
            plt.title(title)
            plt.show()
        else:
            ax.set_title(title)


class ICBHILungSound(LungSound):
    """ Lung sound recording from the ICBHI dataset. """

    def __init__(self, file_path):
        super().__init__(file_path)
        self.text_file = self.wav_file.with_suffix(".txt")

    def load_annotations(self) -> pd.DataFrame:
        columns = ["StartTime", "EndTime", "Crackle", "Wheeze"]
        self.annotations = pd.read_csv(self.text_file, names=columns, delimiter="\t")
        return self.annotations

    @staticmethod
    def parse_metadata(file_path):
        wav_file_name = Path(file_path).name
        parts = wav_file_name.split("_")

        return {
            "FileName": Path(wav_file_name).stem,
            "PatientNumber": parts[0],
            "RecordingIndex": parts[1],
            "ChestLocation": parts[2],
            "AcquisitionMode": parts[3],
            "RecordingEquipment": Path(parts[4]).stem,
        }

    def get_metadata(self):
        return self.parse_metadata(self.wav_file)


class FraiwanLungSound(LungSound):
    """Lung sound recording from the Fraiwan et al. dataset."""
    def __init__(self, file_path):
        super().__init__(file_path)

    @staticmethod
    def parse_metadata(file_path):
        wav_file_name = Path(file_path).name
        stem = Path(wav_file_name).stem

        parts = stem.split("_", 1)
        code_part = parts[0]
        diagnosis = parts[1] if len(parts) > 1 else None

        match = re.match(r"^([BDE])P?(\d+)$", code_part, flags=re.IGNORECASE)
        filter_code = match.group(1).upper() if match else None
        patient_num = f"P{int(match.group(2))}" if match else None

        filter_map = {
            "B": "Bell (20-200 Hz)",
            "D": "Diaphragm (100-500 Hz)",
            "E": "Extended range (50-500 Hz)",
        }

        return {
            "FileName": stem,
            "FilterCode": filter_code,
            "FilterType": filter_map.get(filter_code, "Unknown"),
            "PatientNumber": patient_num,
            "Diagnosis": diagnosis,
        }

    def get_metadata(self):
        return self.parse_metadata(self.wav_file)