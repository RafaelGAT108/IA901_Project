"""
Dataset module for loading and handling lung sound datasets.

Defines the LungSoundDataset class and its subclasses for loading:
    - ICBHI Challenge (2017) respiratory sound database
    - KAUH (2021) respiratory sound database

It also includes functionality for filtering classes, preprocessing,
and splitting the dataset into train/val/test sets.

The dataset is designed to work with audio transforms and feature
extractors defined in the transforms module.
"""

import json
import os
import re
from pathlib import Path
from typing import Any
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from modules.lungsound import LungSoundAudio, LungSoundFeatures
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
    "bronchitis": "Bronchitis",
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


# ============================================================
# ==================== Datasets of Audios ====================
# ============================================================

class LungSoundAudioDataset(Dataset):
    """
    Base dataset for lung sound collections.
    """
    def __init__(
            self,
            root: str | Path,
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | Compose | None = None,
            random_seed: int = 42,
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all". Default is "all".
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included. Default is all classes in the DIAGNOSIS list.
            transform (AudioTransform | Compose | None): Optional transform pipeline to apply to each sample.
            random_seed (int): Random seed for reproducibility when splitting the dataset. Default is 42.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.random_seed = random_seed
        self.classes = [cls_name for cls_name in classes if cls_name in DIAGNOSIS]
        self.labels = {diag: idx for idx, diag in enumerate(self.classes)}
        self.data = None
        self.load_data()
        self.handle_classes()
        if split != "all":
            self.data = self.split_data(
                self.data,
                split,
                train_size=0.8,
                val_size=0.1,
                test_size=0.1,
                seed=self.random_seed
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[LungSoundAudio, int]:
        row = self.data.iloc[idx]
        sample = LungSoundAudio(row["FilePath"])
        label = row["Label"]

        if self.transform is not None:
            sample = self.transform(sample)

        sample.info = row.to_dict()
        return sample, label

    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def handle_classes(self):
        """
        Filter the dataset to only include samples with labels in our DIAGNOSIS mapping.
        """
        # Convert diagnosis to lowercase and map to standard diagnosis names
        self.data["Diagnosis"] = self.data["Diagnosis"].str.lower().map(DIAGNOSIS_MAP)
        # Filter to only include samples with diagnoses in our classes list
        self.data = self.data[self.data["Diagnosis"].isin(self.classes)]
        # Map diagnosis to label indices
        self.data["Label"] = self.data["Diagnosis"].map(self.labels)
        # Reset index after filtering
        self.data.reset_index(drop=True, inplace=True)

    @staticmethod
    def split_data(data: pd.DataFrame, split: str, train_size=0.8, val_size=0.1, test_size=0.1, seed=None) -> pd.DataFrame:
        """
        Split the dataset into train/val/test sets.
        Args:
            data (pd.DataFrame): The full dataset as a pandas DataFrame. Must contain a "Label" column for stratification.
            split (str): Which split to return. Can be "train", "val", "test", or "all".
            train_size (float): Proportion of the dataset to include in the train split. Default is 0.8.
            val_size (float): Proportion of the dataset to include in the validation split. Default is 0.1.
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.1.
            seed (int): Random seed for reproducibility. Default is None.
        Returns:
            A pandas DataFrame containing only the samples for the specified split.
        """
        if split == "all":
            return data

        total_size = train_size + val_size + test_size
        if not abs(total_size - 1.0) < 1e-6:
            raise ValueError(f"train_size + val_size + test_size must equal 1. Got {total_size}.")

        # 1. Split into train and temp (val+test)
        train_data, temp_data = train_test_split(
            data,
            test_size=(1-train_size),
            stratify=data["Label"],
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
        if split == "train":
            return train_data.reset_index(drop=True)
        elif split == "val":
            return val_data.reset_index(drop=True)
        elif split == "test":
            return test_data.reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'all'.")

    def find_wav_files(self):
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
        print(f"  File name: {os.path.basename(sample.file_path)}")
        print(f"  Label: {label} ({label_name})")
        print(f"  Sample rate: {sample.sr}")
        print(f"  Audio shape: {sample.audio.shape}")
        print(f"  Audio dtype: {sample.audio.dtype}")
        print(f"  Audio min value: {sample.audio.min():.3f}")
        print(f"  Audio max value: {sample.audio.max():.3f}")
        print(f"  Duration (s): {sample.duration:.2f}")
        # Plot the waveform
        sample.plot_waveform(title=f"Sample {idx} - Label: {label} ({label_name})")


class ICBHIAudioDataset(LungSoundAudioDataset):
    """ Dataset for the ICBHI Challenge (2017) respiratory sound database."""
    def __init__(
            self,
            root: str | Path,
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | Compose | None = None,
            random_seed: int = 42,
        ):
        self.name = "ICBHI"
        data_dir = os.path.join(root, self.name)
        super().__init__(data_dir, split, classes, transform, random_seed)

    @staticmethod
    def parse_metadata(file_path: str) -> dict:
        """ Parse metadata from a .wav file from the ICBHI dataset. """
        wav_file_stem = Path(file_path).stem
        parts = wav_file_stem.split("_")

        return {
            "PatientId": f"I{parts[0]}",
            "RecordingIndex": f"{parts[1]}",
            "ChestLocation": f"{parts[2]}",
            "AcquisitionMode": f"{parts[3]}",
            "RecordingEquipment": f"{parts[4]}",
        }

    def load_data(self) -> pd.DataFrame:
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
        for wav_file in self.find_wav_files():
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


class KAUHAudioDataset(LungSoundAudioDataset):
    """ Dataset for the KAUH (2021) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | Compose | None = None,
            random_seed: int = 42
        ):
        self.name = "KAUH"
        data_dir = os.path.join(root, self.name)
        super().__init__(data_dir, split, classes, transform, random_seed)

    @staticmethod
    def parse_metadata(file_path: str) -> dict:
        """ Parse metadata from a .wav file from the KAUH dataset. """
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

        return {
            "Diagnosis": diagnosis,
            "FilterCode": filter_code,
            "PatientId": patient_num,
            "Sound type": sound_type,
            "Location": location,
            "Age": float(age),
            "Sex": sex,
        }

    def load_data(self) -> pd.DataFrame:
        """ Load data from the KAUH dataset. """
        records = []
        for wav_file in self.find_wav_files():
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


class CombinedAudioDataset(LungSoundAudioDataset):
    """ Dataset that combines ICBHI and KAUH before splitting. """
    def __init__(
            self,
            root: str | Path,
            split: str,
            classes: list[str],
            transform: AudioTransform | Compose | None = None,
            random_seed: int = 42
        ):
        self.name = "Combined_ICBHI_KAUH"
        super().__init__(root, split, classes, transform, random_seed)

    def load_data(self) -> pd.DataFrame:
        """ Load and combine data from both ICBHI and KAUH datasets. """
        icbhi_data = ICBHIAudioDataset(self.root, "all", self.classes).data
        kauh_data = KAUHAudioDataset(self.root, "all", self.classes).data
        self.data = pd.concat([icbhi_data, kauh_data], ignore_index=True)
        return self.data


# ============================================================ #
# =================== Datasets of Features =================== #
# ============================================================ #

class LungSoundFeaturesDataset(Dataset):
    """
    Dataset for lung sound features extracted from audio files.
    """
    def __init__(
            self,
            root: str | Path,
            split: str,
            feature_extractor: str,
            classes: list[str] = DIAGNOSIS,
            random_seed: int = 42
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all".
            feature_extractor (str): Name of the feature extractor used to generate the features. This should correspond to a subdirectory in the preprocessed data directory.
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included. Default is all classes in the DIAGNOSIS list.
            random_seed (int): Random seed for reproducibility when splitting the dataset. Default is 42.
        """
        self.root = Path(root)
        self.split = split
        self.feature_extractor = feature_extractor
        self.random_seed = random_seed
        self.classes = [cls_name for cls_name in classes if cls_name in DIAGNOSIS]
        self.labels = {diag: idx for idx, diag in enumerate(self.classes)}
        self.data = None
        self.load_data()
        self.handle_classes()
        if split != "all":
            self.data = LungSoundAudioDataset.split_data(
                self.data,
                split,
                train_size=0.8,
                val_size=0.1,
                test_size=0.1,
                seed=self.random_seed
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[LungSoundFeatures, int]:
        row = self.data.iloc[idx]
        file_path = row["FilePath"]
        label = row["Label"]
        sample = LungSoundFeatures(file_path)
        sample.info = row.to_dict()
        return sample, label

    def get_preprocessing(self) -> Any:
        """ Load the transformations from the JSON file saved during preprocessing. """
        data_dir = self.root / self.feature_extractor
        json_path = data_dir / "preprocessing.json"
        with open(json_path, "r") as f:
            preprocessing = json.load(f)
        return preprocessing

    def get_plot_params(self) -> dict:
        """ Get the parameters to use when plotting the features with librosa.display.specshow. """
        feature_extractor_info = self.preprocessing.get("feature_extractor", {})
        extractor_name = next(iter(feature_extractor_info.keys()), None)
        if extractor_name is None:
            return {}
        return feature_extractor_info[extractor_name].get("plot_params", {})

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the CSV file and construct the file paths.
        """
        self.data = pd.read_csv(self.root / "data.csv")
        data_dir = self.root / self.feature_extractor
        self.data["FilePath"] = self.data.apply(lambda row: str(data_dir / row["Diagnosis"] / f"{row['FileName']}"), axis=1)
        self.preprocessing = self.get_preprocessing()
        return self.data

    def handle_classes(self):
        """
        Filter the dataset to only include samples with labels in our DIAGNOSIS mapping.
        """
        # Filter to only include samples with diagnoses in our classes list
        self.data = self.data[self.data["Diagnosis"].isin(self.classes)]
        # Map diagnosis to label indices
        self.data["Label"] = self.data["Diagnosis"].map(self.labels)
        # Reset index after filtering
        self.data.reset_index(drop=True, inplace=True)

    def unit_test(self, idx=0):
        """
        Run a simple unit test to verify data.
        """
        sample, label = self[idx]
        label_name = self.classes[label]
        file_name = self.data.iloc[idx]["FileName"]
        print(f"Sample {idx}:")
        print(f"  File name: {file_name}")
        print(f"  Label: {label} ({label_name})")
        print(f"  Sample rate: {sample.sr}")
        print(f"  Feature shape: {sample.features.shape}")
        print(f"  Feature dtype: {sample.features.dtype}")
        print(f"  Feature min value: {sample.features.min():.3f}")
        print(f"  Feature max value: {sample.features.max():.3f}")
        # Plot the features
        plot_params = self.get_plot_params()
        sample.plot_features(title=file_name, **plot_params)


class ICBHIFeaturesDataset(LungSoundFeaturesDataset):
    """ Dataset for features extracted from the ICBHI Challenge (2017) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            split: str,
            feature_extractor: str,
            classes: list[str] = DIAGNOSIS,
            random_seed: int = 42
        ):
        self.name = "ICBHI"
        data_dir = os.path.join(root, self.name)
        super().__init__(data_dir, split, feature_extractor, classes, random_seed)


class KAUHFeaturesDataset(LungSoundFeaturesDataset):
    """ Dataset for features extracted from the KAUH (2021) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            split: str,
            feature_extractor: str,
            classes: list[str] = DIAGNOSIS,
            random_seed: int = 42
        ):
        self.name = "KAUH"
        data_dir = os.path.join(root, self.name)
        super().__init__(data_dir, split, feature_extractor, classes, random_seed)


class CombinedFeaturesDataset(LungSoundFeaturesDataset):
    """ Dataset that combines features from both ICBHI and KAUH before splitting. """
    def __init__(
            self,
            root: str | Path,
            split: str,
            feature_extractor: str,
            classes: list[str] = DIAGNOSIS,
            random_seed: int = 42
        ):
        self.name = "Combined_ICBHI_KAUH"
        super().__init__(root, split, feature_extractor, classes, random_seed)

    def load_data(self) -> pd.DataFrame:
        """ Load and combine data from both ICBHI and KAUH datasets. """
        icbhi = ICBHIFeaturesDataset(self.root, "all", self.feature_extractor, self.classes)
        kauh = KAUHFeaturesDataset(self.root, "all", self.feature_extractor, self.classes)
        self.data = pd.concat([icbhi.data, kauh.data], ignore_index=True)
        self.preprocessing_icbhi = icbhi.preprocessing
        self.preprocessing_kauh = kauh.preprocessing
        self.preprocessing = {**self.preprocessing_icbhi, **self.preprocessing_kauh}
        return self.data