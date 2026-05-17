"""
Dataset module for loading and handling lung sound datasets.

Defines the LungSoundDataset class and its subclasses for loading:
    - ICBHI Challenge (2017) respiratory sound database
    - Fraiwan et al. (2021) respiratory sound database

It also includes functionality for filtering classes, preprocessing,
and splitting the dataset into train/val/test sets.

The dataset is designed to work with audio transforms and feature
extractors defined in the transforms module.
"""

import os
import re
from pathlib import Path
import librosa
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from modules.audio import LungSound
from modules.transforms import *


# The standardized set of diagnosis classes we will use across both datasets
DIAGNOSIS = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "Bronchitis",
    "COPD",
    "Healthy",
    "Heart Failure",
    "LRTI",
    "Lung Fibrosis",
    "Pleural Effusion",
    "Pneumonia",
    "URTI",
]

# Map various diagnosis labels from the datasets to our standardized set of classes
DIAGNOSIS_MAP = {
    "asthma": "Asthma",
    "bronchiectasis": "Bronchiectasis",
    "bronchiolitis": "Bronchiolitis",
    "bron": "Bronchitis",
    "copd": "COPD",
    "healthy": "Healthy",
    "n": "Healthy",
    "heart failure": "Heart Failure",
    "lrti": "LRTI",
    "lung fibrosis": "Lung Fibrosis",
    "pleural effusion": "Pleural Effusion",
    "plueral effusion": "Pleural Effusion",
    "pneumonia": "Pneumonia",
    "urti": "URTI",
}


class LungSoundDataset(Dataset):
    """
    Base dataset for lung sound collections.
    """
    def __init__(
            self,
            root: str | Path,
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | FeatureExtractor | DatasetTransform | Compose | None = None,
            load_data_on_init: bool = True
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all". Default is "all".
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included. Default is all classes in the DIAGNOSIS list.
            transform (AudioTransform | FeatureExtractor | DatasetTransform | Compose | None): Optional transform pipeline to apply to each sample or the entire dataset.
            load_data_on_init (bool): Whether to load the data immediately upon initialization. Set to False if you want to delay loading.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.classes = [cls_name for cls_name in classes if cls_name in DIAGNOSIS]
        self.labels = {diag: idx for idx, diag in enumerate(self.classes)}
        self.data = None
        self.windowing = False
        if load_data_on_init:
            self._load_data()
            self._handle_classes()
            self._handle_windowing()
            self._split_data(train_size=0.8, val_size=0.1, test_size=0.1, seed=42)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[LungSound, str]:
        row = self.data.iloc[idx]
        sample = LungSound(row["FilePath"])
        label = row["Label"]

        if self.windowing:
            sample.window_start = row["Start"]
            sample.window_end = row["End"]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label

    def _load_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _handle_classes(self):
        """
        Filter the dataset to only include samples with labels in our DIAGNOSIS mapping.
        """
        # Convert diagnosis to lowercase and map to standard diagnosis names
        self.data["Diagnosis"] = self.data["Diagnosis"].str.lower().map(DIAGNOSIS_MAP)
        # Filter to only include samples with diagnoses in our classes list
        self.data = self.data[self.data["Diagnosis"].isin(self.classes)]
        # Update attributes to keep only the classes actually present in the dataset
        self.classes = sorted(self.data["Diagnosis"].unique().tolist())
        self.labels = {diag: idx for idx, diag in enumerate(self.classes)}
        # Map diagnosis to label indices
        self.data["Label"] = self.data["Diagnosis"].map(self.labels)
        # Reset index after filtering
        self.data.reset_index(drop=True, inplace=True)

    def _handle_windowing(self) -> None:
        """
        Handle windowing to the dataset by expanding the DataFrame with new rows for each window.
        """
        self.windowing = False
        if self.transform is None:
            return
        if isinstance(self.transform, Compose):
            # Check if there is a Window transform in the preprocessing pipeline
            for step in self.transform.transforms:
                if isinstance(step, Window):
                    self.data = step.modify_dataframe(self.data)
                    self.windowing = True
                    return
        if isinstance(self.transform, Window):
            self.data = self.transform.modify_dataframe(self.data)
            self.windowing = True
            return
        # No windowing transform found in the preprocessing pipeline

    def _split_data(self, train_size=0.8, val_size=0.1, test_size=0.1, seed=42) -> None:
        """
        Split the dataset into train/val/test sets.
        """
        if self.split == "all":
            return

        total_size = train_size + val_size + test_size
        if not abs(total_size - 1.0) < 1e-6:
            raise ValueError(f"train_size + val_size + test_size must equal 1. Got {total_size}.")

        # 1. Split into train and temp (val+test)
        train_data, temp_data = train_test_split(
            self.data,
            test_size=(1-train_size),
            stratify=self.data["Label"],
            shuffle=True,
            random_state=seed
        )
        # 2. Split temp into val and test
        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_size / (test_size + val_size),
            stratify=temp_data["Label"],
            shuffle=True,
            random_state=seed
        )
        # 3. Assign the appropriate split to self.data
        if self.split == "train":
            self.data = train_data.reset_index(drop=True)
        elif self.split == "val":
            self.data = val_data.reset_index(drop=True)
        elif self.split == "test":
            self.data = test_data.reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {self.split}. Must be 'train', 'val', 'test', or 'all'.")

    def _find_wav_files(self):
        """
        Recursively find all .wav files in the root directory.
        """
        return sorted(self.root.rglob("*.wav"))

    def unit_test(self, idx=0):
        """
        Run a simple unit test to verify data.
        """
        sample, label = self[idx]
        label_name = self.classes[label]
        print(f"Sample {idx}:")
        print(f"  File name: {os.path.basename(sample.wav_file)}")
        print(f"  Label: {label} ({label_name})")
        print(f"  Sample rate: {sample.sr}")
        print(f"  Audio shape: {sample.audio.shape}")
        # Plot the waveform
        sample.plot_waveform(title=f"Sample {idx} - Label: {label} ({label_name})")
        # Plot the features if they exist
        if sample.has_features:
            print(f"  Features shape: {sample.features.shape}")
            print(f"  Features dimensions: {sample.features.ndim}")
            print(f"  Features dtype: {sample.features.dtype}")
            sample.plot_features(title=f"Sample {idx} - Label: {label} ({label_name})")


class ICBHIDataset(LungSoundDataset):
    """ Dataset for the ICBHI Challenge (2017) respiratory sound database."""
    def __init__(
            self,
            root: str | Path,
            split: str,
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | FeatureExtractor | DatasetTransform | Compose | None = None,
            load_data_on_init: bool = True
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all".
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included.
            transform (AudioTransform | FeatureExtractor | DatasetTransform | Compose | None): Optional transform pipeline to apply to each sample or dataset.
            load_data_on_init (bool): Whether to load the data immediately upon initialization. Set to False if you want to delay loading.
        """
        self.name = "icbhi"
        data_dir = os.path.join(root, "ICBHI_final_database")
        super().__init__(data_dir, split, classes, transform, load_data_on_init)

    @staticmethod
    def parse_metadata(file_path: str) -> dict:
        """ Parse metadata from a .wav file from the ICBHI dataset. """
        wav_file_stem = Path(file_path).stem
        parts = wav_file_stem.split("_")
        duration = librosa.get_duration(path=file_path)

        return {
            "PatientId": f"I{parts[0]}",
            "RecordingIndex": f"{parts[1]}",
            "ChestLocation": f"{parts[2]}",
            "AcquisitionMode": f"{parts[3]}",
            "RecordingEquipment": f"{parts[4]}",
            "AudioDuration": duration,
        }

    def _load_data(self) -> pd.DataFrame:
        """ Load data from the ICBHI dataset. """
        # Diagnosis file
        diagnosis_file = os.path.join(self.root, "ICBHI_Challenge_diagnosis.txt")
        diagnosis_df = pd.read_csv(
            diagnosis_file,
            names=["PatientNumber", "Diagnosis"],
            delimiter="\t",
            dtype={"PatientNumber": str, "Diagnosis": str},
        )
        diagnosis_map = dict(zip(diagnosis_df["PatientNumber"], diagnosis_df["Diagnosis"]))

        # Metadata file
        demographic_file = os.path.join(self.root, "ICBHI_Challenge_demographic_information.txt")
        demographic_df = pd.read_csv(
            demographic_file,
            names=["PatientNumber", "Age", "Sex", "AdultBMI (kg/m2)", "ChildWheight (kg)", "ChildHeight (cm)"],
            delimiter="\t",
            na_values=["NA"],
            keep_default_na=False,
            dtype={
                "PatientNumber": str,
                "Age": float,
                "Sex": str,
                "AdultBMI (kg/m2)": float,
                "ChildWheight (kg)": float,
                "ChildHeight (cm)": float,
            },
        )
        demographic_map = demographic_df.set_index("PatientNumber").to_dict(orient="index")

        records = []
        for wav_file in self._find_wav_files():
            metadata = self.parse_metadata(wav_file)
            patient_number = metadata["PatientId"][1:]
            demographic_info = demographic_map.get(patient_number, {})
            records.append(
                {
                    "Source": self.name,
                    "FilePath": str(wav_file),
                    "Diagnosis": diagnosis_map.get(patient_number, "Unknown"),
                    **demographic_info,
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)
        return self.data


class FraiwanDataset(LungSoundDataset):
    """ Dataset for the Fraiwan et al. (2021) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            split: str,
            classes: list[str],
            transform: AudioTransform | FeatureExtractor | DatasetTransform | Compose | None = None,
            load_data_on_init: bool = True
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all".
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included.
            transform (AudioTransform | FeatureExtractor | DatasetTransform | Compose | None): Optional transform pipeline to apply to each sample or dataset.
            load_data_on_init (bool): Whether to load the data immediately upon initialization. Set to False if you want to delay loading.
        """
        self.name = "fraiwan"
        data_dir = os.path.join(root, "fraiwan")
        super().__init__(data_dir, split, classes, transform, load_data_on_init)

    @staticmethod
    def parse_metadata(file_path: str) -> dict:
        """ Parse metadata from a .wav file from the Fraiwan dataset. """
        wav_file_stem = Path(file_path).stem

        parts = wav_file_stem.split("_", 1)
        code_part = parts[0]
        metadata_part = parts[1]

        metadata_values = [part.strip() for part in metadata_part.split(",", 4)]
        if len(metadata_values) == 5:
            diagnosis, sound_type, location, age, sex = metadata_values
        else:
            return {}

        match = re.match(r"^([BDE])P?(\d+)$", code_part, flags=re.IGNORECASE)
        filter_code = match.group(1).upper() if match else None
        patient_num = f"P{int(match.group(2))}" if match else None
        duration = librosa.get_duration(path=file_path)

        return {
            "Diagnosis": diagnosis,
            "FilterCode": filter_code,
            "PatientId": patient_num,
            "Sound type": sound_type,
            "Location": location,
            "Age": float(age),
            "Sex": sex,
            "AudioDuration": duration,
        }

    def _load_data(self) -> pd.DataFrame:
        """ Load data from the Fraiwan dataset. """
        records = []
        for wav_file in self._find_wav_files():
            metadata = self.parse_metadata(wav_file)
            records.append(
                {
                    "Source": self.name,
                    "FilePath": str(wav_file),
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)
        return self.data


class CombinedLungSoundDataset(LungSoundDataset):
    """ Dataset that combines ICBHI and Fraiwan before splitting. """
    def __init__(
            self,
            root: str | Path,
            split: str,
            classes: list[str],
            transform: AudioTransform | FeatureExtractor | DatasetTransform | Compose | None = None,
            load_data_on_init: bool = True
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all".
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included.
            transform (AudioTransform | FeatureExtractor | DatasetTransform | Compose | None): Optional transform pipeline to apply to each sample or dataset.
            load_data_on_init (bool): Whether to load the data immediately upon initialization. Set to False if you want to delay loading.
        """
        super().__init__(root, split, classes, transform, load_data_on_init)

    def _load_data(self) -> pd.DataFrame:
        """ Load and combine data from both ICBHI and Fraiwan datasets. """
        icbhi_dataset = ICBHIDataset(self.root, "all", self.classes, load_data_on_init=False)
        icbhi_data = icbhi_dataset._load_data()
        fraiwan_dataset = FraiwanDataset(self.root, "all", self.classes, load_data_on_init=False)
        fraiwan_data = fraiwan_dataset._load_data()
        self.data = pd.concat([icbhi_data, fraiwan_data], ignore_index=True)
        return self.data