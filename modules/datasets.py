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
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
from torch.utils.data import Dataset
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

class AudioDataset(Dataset):
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
            sample_limit: int | None = None
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all". Default is "all".
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included. Default is all classes in the DIAGNOSIS list.
            transform (AudioTransform | Compose | None): Optional transform pipeline to apply to each sample.
            random_seed (int): Random seed for reproducibility when splitting the dataset. Default is 42.
            sample_limit (int | None): Maximum number of samples per class to include in the dataset. If None, include all samples. Default is None.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.random_seed = random_seed
        self.sample_limit = sample_limit
        self.classes = [cls_name for cls_name in classes if cls_name in DIAGNOSIS]
        self.labels = {diag: idx for idx, diag in enumerate(self.classes)}
        self.data = None
        self.load_data()
        self.handle_classes()
        if self.sample_limit is not None:
            self.data = self.apply_sample_limit(
                self.data, self.sample_limit, self.random_seed
            )
        if split != "all":
            self.data = self.split_data(
                self.data,
                split,
                train_size=0.8,
                val_size=0.1,
                test_size=0.1,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[LungSoundAudio, int]:
        row = self.data.iloc[idx]
        file_path = self.root / row["FilePath"]
        sample = LungSoundAudio(file_path)
        label = row["Label"]

        if self.transform is not None:
            sample = self.transform(sample)

        sample.info = row.to_dict()
        return sample, label

    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError

    @staticmethod
    def apply_sample_limit(data: pd.DataFrame, sample_limit: int, random_seed: int) -> pd.DataFrame:
        """
        Apply a sample limit per class to the dataset.
        Args:
            data (pd.DataFrame): The full dataset as a pandas DataFrame. Must contain a "Diagnosis" column.
            sample_limit (int): Maximum number of samples per class to include in the dataset.
            random_seed (int): Random seed for reproducibility when sampling.
        Returns:
            pd.DataFrame: The dataset with the sample limit applied.
        """
        dfs = []
        for i, group in data.groupby("Diagnosis"):
            if len(group) <= sample_limit:
                dfs.append(group)
            else:
                unique_patients = pd.Series(group["PatientId"].unique())
                if sample_limit <= len(unique_patients):
                    # Choose a subset of unique patients
                    selected_patients = unique_patients.sample(n=sample_limit, random_state=random_seed).tolist()
                    # Filter the group to include only the selected patients
                    df_filtered = group[group["PatientId"].isin(selected_patients)]
                    # Then, sample 1 random sample from each of the selected patients
                    df_i = df_filtered.groupby("PatientId").sample(n=1, random_state=random_seed)
                else:
                    # First, choose 1 random sample for each unique patient
                    df_i = group.groupby("PatientId").sample(n=1, random_state=random_seed)
                    # Then, randomly sample additional patients until we reach the sample limit
                    df_remaining = group[~group.index.isin(df_i.index)]
                    sample_limit_remaining = sample_limit - len(df_i)
                    df_additional = df_remaining.sample(n=sample_limit_remaining, random_state=random_seed)
                    df_i = pd.concat([df_i, df_additional], ignore_index=True)
                dfs.append(df_i)

        return pd.concat(dfs, ignore_index=True)

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
    def split_data(data: pd.DataFrame, split: str, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1) -> pd.DataFrame:
        """
        Split the dataset into train/val/test sets.
        It performs a group-based stratified split based on the "PatientId" to ensure that:
        1. No patient appears in more than one split (train/val/test) to prevent data leakage.
        2. Each split contains samples from all classes to avoid empty classes in any partition.
        3. The proportions of train/val/test are approximately maintained for each class.
            It uses a greedy algorithm to assign patients to splits while trying to balance the class distributions.

        Args:
            data (pd.DataFrame): The full dataset as a pandas DataFrame. Must contain 'Label' and 'PatientId' columns.
            split (str): Which split to return. Can be "train", "val", "test", or "all".
            train_size (float): Proportion of the dataset to include in the train split. Default is 0.8.
            val_size (float): Proportion of the dataset to include in the validation split. Default is 0.1.
            test_size (float): Proportion of the dataset to include in the test split. Default is 0.1.
        Returns:
            A pandas DataFrame containing only the samples for the specified split.
        """
        if split == "all":
            return data

        total_size = train_size + val_size + test_size
        if not abs(total_size - 1.0) < 1e-6:
            raise ValueError(f"train_size + val_size + test_size must equal 1. Got {total_size}.")

        if "PatientId" not in data.columns:
            raise KeyError("The 'PatientId' column is required to perform group-based splitting without data leakage.")

        # Map sample counts and medical labels to each unique patient ID
        patient_counts = data.groupby("PatientId").size().to_dict()             # patient_counts = {PatientId: count for PatientID}
        patient_labels = data.groupby("PatientId")["Label"].first().to_dict()   # patient_labels = {PatientId: Label}
        
        # Group patient IDs by their diagnosis class for stratification
        all_classes = sorted(data["Label"].unique())
        patients_by_class = {c: [] for c in all_classes}
        for p, label in patient_labels.items():
            patients_by_class[label].append(p)

        # Initialize sets to store the isolated patient IDs assigned to each partition
        train_patients, val_patients, test_patients = set(), set(), set()

        # ------
        # Step 1: Initial allocation to ensure no empty classes in any split
        # ------
        # Allocate at least one patient per class to each split to ensure no empty classes.
        # Patients within each class are sorted by sample size in descending order for determinism.
        for label, p_list in patients_by_class.items():
            p_list.sort(key=lambda p: patient_counts[p], reverse=True)

            if len(p_list) >= 3:
                train_patients.add(p_list.pop(0))  # Largest patient is assigned to Train
                val_patients.add(p_list.pop(0))    # Second largest is assigned to Validation
                test_patients.add(p_list.pop(0))   # Third largest is assigned to Test
            else:
                raise ValueError(
                    f"Class {label} has only {len(p_list)} patients. "
                    f"Cannot distribute across 3 splits without causing patient leakage."
                )

        # ------
        # Step 2: Greedy algorithm to assign remaining patients while balancing class distributions
        # ------
        # Collect all remaining unallocated patients and sort them by descending sample size
        remaining_patients = [p for p_list in patients_by_class.values() for p in p_list]
        remaining_patients.sort(key=lambda p: patient_counts[p], reverse=True)

        # Distribute unallocated patients based on the target proportions of their specific class
        for p in remaining_patients:
            p_cls = patient_labels[p]   # Class of the current patient

            # Count the current total number of samples for this class already allocated to each split
            current_train_cls = sum(patient_counts[pt] for pt in train_patients if patient_labels[pt] == p_cls)
            current_val_cls = sum(patient_counts[pv] for pv in val_patients if patient_labels[pv] == p_cls)
            current_test_cls = sum(patient_counts[ptest] for ptest in test_patients if patient_labels[ptest] == p_cls)

            total_cls_allocated = current_train_cls + current_val_cls + current_test_cls

            # Calculate the ideal target sizes for this class based on the current subset volume
            target_train_cls = total_cls_allocated * train_size
            target_val_cls = total_cls_allocated * val_size
            target_test_cls = total_cls_allocated * test_size

            # Determine the current sample deficits for this class in each split
            deficit_train = target_train_cls - current_train_cls
            deficit_val = target_val_cls - current_val_cls
            deficit_test = target_test_cls - current_test_cls

            # Assign the current patient to the split displaying the largest specific class deficit
            max_deficit = max(deficit_train, deficit_val, deficit_test)
            if max_deficit == deficit_train:
                train_patients.add(p)
            elif max_deficit == deficit_val:
                val_patients.add(p)
            else:
                test_patients.add(p)

        # ------
        # Step 3: Return the appropriate split based on the assigned patient IDs
        # ------
        if split == "train":
            return data[data["PatientId"].isin(train_patients)].reset_index(drop=True)
        elif split == "val":
            return data[data["PatientId"].isin(val_patients)].reset_index(drop=True)
        elif split == "test":
            return data[data["PatientId"].isin(test_patients)].reset_index(drop=True)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', 'test', or 'all'.")

    def find_wav_files(self):
        """
        Recursively find all .wav files in the root directory.
        """
        return sorted((self.root / self.name).rglob("*.wav"))

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


class ICBHIAudioDataset(AudioDataset):
    """ Dataset for the ICBHI Challenge (2017) respiratory sound database."""
    def __init__(
            self,
            root: str | Path,
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | Compose | None = None,
            random_seed: int = 42,
            sample_limit: int | None = None,
        ):
        self.name = "ICBHI"
        super().__init__(root, split, classes, transform, random_seed, sample_limit)

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
        # In case it's a preprocessed dataset with a metadata.csv file
        metadata_file = os.path.join(self.root, self.name, "metadata.csv")
        if os.path.exists(metadata_file):
            self.data = pd.read_csv(metadata_file)
            return self.data

        # Diagnosis file
        diagnosis_file = os.path.join(self.root, self.name, "ICBHI_Challenge_diagnosis.txt")
        diagnosis_df = pd.read_csv(
            diagnosis_file,
            names=["PatientNumber", "Diagnosis"],
            delimiter="\t",
            dtype={"PatientNumber": str, "Diagnosis": str},
        )
        diagnosis_map = dict(zip(diagnosis_df["PatientNumber"], diagnosis_df["Diagnosis"]))

        # Metadata file
        demographic_file = os.path.join(self.root, self.name, "ICBHI_Challenge_demographic_information.txt")
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
                    "FilePath": os.path.relpath(wav_file, self.root),
                    "Diagnosis": diagnosis_map.get(patient_number, "Unknown"),
                    **demographic_info,
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)
        return self.data


class KAUHAudioDataset(AudioDataset):
    """ Dataset for the KAUH (2021) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | Compose | None = None,
            random_seed: int = 42,
            sample_limit: int | None = None,
        ):
        self.name = "KAUH"
        super().__init__(root, split, classes, transform, random_seed, sample_limit)

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
            "ChestLocation": location,
            "Age": float(age),
            "Sex": sex,
        }

    def load_data(self) -> pd.DataFrame:
        """ Load data from the KAUH dataset. """
        # In case it's a preprocessed dataset with a metadata.csv file
        metadata_file = os.path.join(self.root, self.name, "metadata.csv")
        if os.path.exists(metadata_file):
            self.data = pd.read_csv(metadata_file)
            return self.data

        records = []
        for wav_file in self.find_wav_files():
            metadata = self.parse_metadata(wav_file)
            records.append(
                {
                    "Source": self.name,
                    "FilePath": os.path.relpath(wav_file, self.root),
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)
        return self.data


class CombinedAudioDataset(AudioDataset):
    """ Dataset that combines ICBHI and KAUH before splitting. """
    def __init__(
            self,
            root: str | Path,
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: AudioTransform | Compose | None = None,
            random_seed: int = 42,
            sample_limit: int | None = None,
        ):
        self.name = "Combined_ICBHI_KAUH"
        super().__init__(root, split, classes, transform, random_seed, sample_limit)

    def load_data(self) -> pd.DataFrame:
        """ Load and combine data from both ICBHI and KAUH datasets. """
        icbhi_data = ICBHIAudioDataset(self.root, "all", self.classes).data
        kauh_data = KAUHAudioDataset(self.root, "all", self.classes).data
        self.data = pd.concat([icbhi_data, kauh_data], ignore_index=True)
        return self.data


# ============================================================ #
# =================== Datasets of Features =================== #
# ============================================================ #

class FeaturesDataset(Dataset):
    """
    Dataset for lung sound features extracted from audio files.
    """
    def __init__(
            self,
            root: str | Path,
            feature_extractor: str | list[str],
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: FeatureTransform | Compose | None = None,
            random_seed: int = 42,
            sample_limit: int | None = None
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            feature_extractor (str | list[str]): Name(s) of the feature extractor(s) used to generate the features. This should correspond to a subdirectory in the preprocessed data directory.
            split (str): Dataset split. Can be "train", "val", "test", or "all". Default is "all".
            classes (list[str]): List of classes to include in the dataset. Only samples with these diagnoses will be included. Default is all classes in the DIAGNOSIS list.
            transform (FeatureTransform | Compose | None): Optional transform pipeline to apply to each sample's features.
            random_seed (int): Random seed for reproducibility when splitting the dataset. Default is 42.
            sample_limit (int | None): Maximum number of samples per class to include in the dataset. If None, include all samples. Default is None.
        """
        self.root = Path(root)
        self.feature_extractor = feature_extractor
        self.split = split
        self.transform = transform
        self.random_seed = random_seed
        self.sample_limit = sample_limit
        self.classes = [cls_name for cls_name in classes if cls_name in DIAGNOSIS]
        self.labels = {diag: idx for idx, diag in enumerate(self.classes)}
        self.data = None
        self.load_data()
        self.handle_classes()
        if self.sample_limit is not None:
            self.data = AudioDataset.apply_sample_limit(
                self.data, self.sample_limit, self.random_seed
            )
        if split != "all":
            self.data = AudioDataset.split_data(
                self.data,
                split,
                train_size=0.8,
                val_size=0.1,
                test_size=0.1,
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[LungSoundFeatures, int]:
        row = self.data.iloc[idx]
        label = row["Label"]

        if isinstance(self.feature_extractor, list):
            # If it's a list of extractors, we need to stack the features from each extractor
            sample = self.stack_features(idx, self.feature_extractor)
        else:
            # If it's a single extractor, we can load the features directly
            source = row["Source"]
            diagnosis = row["Diagnosis"]
            file_name = row["FileName"]
            file_path = self.root / source / self.feature_extractor / diagnosis / file_name
            sample = LungSoundFeatures(file_path)

        if self.transform is not None:
            sample = self.transform(sample)

        sample.info = row.to_dict()
        return sample, label

    def stack_features(self, idx: int, extractors: list[str]) -> LungSoundFeatures:
        """
        Stack features from multiple extractors along the channel dimension.
        Args:
            idx (int): Index of the sample to load and stack features for.
            extractors (list[str]): List of feature extractor names to stack. Each should correspond to a subdirectory in the preprocessed data directory.
        Returns:
            LungSoundFeatures: The stacked features for the sample at the given index.
        """
        sample = self.data.iloc[idx]
        source = sample["Source"]
        diagnosis = sample["Diagnosis"]
        file_name = sample["FileName"]
        all_features = []
        sr = set()
        for extractor in extractors:
            file_path = self.root / source / extractor / diagnosis / file_name
            features_sample = LungSoundFeatures(file_path)
            all_features.append(features_sample.features)
            sr.add(features_sample.sr)
        # Check that all sample rates are the same
        if len(sr) > 1:
            raise ValueError(f"All feature extractors must have the same sample rate. Got {sr}.")
        # Stack the features along the channel dimension
        features = np.stack(all_features, axis=-1)
        stacked_sample = LungSoundFeatures(features=features, sr=sr.pop())
        return stacked_sample

    def get_preprocessing(self) -> Any:
        if isinstance(self.feature_extractor, list):
            preprocessing = {}
            for extractor in self.feature_extractor:
                data_dir = self.root / self.name / extractor
                json_path = data_dir / "features_preprocessing.json"
                with open(json_path, "r") as f:
                    extractor_preprocessing = json.load(f)
                preprocessing[extractor] = extractor_preprocessing
            return preprocessing
        else:
            data_dir = self.root / self.name / self.feature_extractor
            json_path = data_dir / "features_preprocessing.json"
            with open(json_path, "r") as f:
                preprocessing = json.load(f)
            return preprocessing

    def get_plot_params(self) -> dict:
        """ Get the parameters to use when plotting the features with librosa.display.specshow. """
        if isinstance(self.feature_extractor, list):
            plot_params = {}
            for extractor in self.preprocessing.keys():
                extractor_info = self.preprocessing[extractor]["feature_extractor"][extractor]
                plot_params[extractor] = extractor_info.get("plot_params", {})
        else:
            extractor_info = self.preprocessing["feature_extractor"][self.feature_extractor]
            plot_params = extractor_info.get("plot_params", {})
        return plot_params

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the CSV file and construct the file paths.
        """
        self.data = pd.read_csv(self.root / self.name / "metadata.csv")
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
        print(f"  Feature extractor: {self.feature_extractor}")
        print(f"  Feature shape: {sample.features.shape}")
        print(f"  Feature dtype: {sample.features.dtype}")
        print(f"  Feature min value: {sample.features.min():.3f}")
        print(f"  Feature max value: {sample.features.max():.3f}")
        # Plot the features
        plot_params = self.get_plot_params()
        if isinstance(self.feature_extractor, list):
            # If it's a stack, we need to plot each feature in the stack separately
            num_features = len(plot_params.keys())
            fig, axes = plt.subplots(1, num_features, figsize=(5 * num_features, 4), constrained_layout=True)
            axes = axes.flatten()
            for i, (stack_feature, params) in enumerate(plot_params.items()):
                features = sample.features[:, :, i]
                sr = sample.sr
                sample_channel = LungSoundFeatures(features=features, sr=sr)
                sample_channel.plot_features(title=f"{stack_feature}", ax=axes[i], **params)
            plt.suptitle(f"{file_name}")
            plt.show()
        else:
            # If it's a single extractor, we can plot the features directly
            sample.plot_features(title=f"{file_name}", **plot_params)


class ICBHIFeaturesDataset(FeaturesDataset):
    """ Dataset for features extracted from the ICBHI Challenge (2017) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            feature_extractor: str | list[str],
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: FeatureTransform | Compose | None = None,
            random_seed: int = 42.,
            sample_limit: int | None = None,
        ):
        self.name = "ICBHI"
        super().__init__(root, feature_extractor, split, classes, transform, random_seed, sample_limit)


class KAUHFeaturesDataset(FeaturesDataset):
    """ Dataset for features extracted from the KAUH (2021) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            feature_extractor: str | list[str],
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: FeatureTransform | Compose | None = None,
            random_seed: int = 42,
            sample_limit: int | None = None,
        ):
        self.name = "KAUH"
        super().__init__(root, feature_extractor, split, classes, transform, random_seed, sample_limit)


class CombinedFeaturesDataset(FeaturesDataset):
    """ Dataset that combines features from both ICBHI and KAUH before splitting. """
    def __init__(
            self,
            root: str | Path,
            feature_extractor: str | list[str],
            split: str = "all",
            classes: list[str] = DIAGNOSIS,
            transform: FeatureTransform | Compose | None = None,
            random_seed: int = 42,
            sample_limit: int | None = None,
        ):
        self.name = "Combined_ICBHI_KAUH"
        super().__init__(root, feature_extractor, split, classes, transform, random_seed, sample_limit)

    def load_data(self) -> pd.DataFrame:
        """ Load and combine data from both ICBHI and KAUH datasets. """
        icbhi = ICBHIFeaturesDataset(self.root, self.feature_extractor, "all", self.classes)
        kauh = KAUHFeaturesDataset(self.root, self.feature_extractor, "all", self.classes)
        self.data = pd.concat([icbhi.data, kauh.data], ignore_index=True)
        self.preprocessing_icbhi = icbhi.preprocessing
        self.preprocessing_kauh = kauh.preprocessing
        self.preprocessing = {**self.preprocessing_icbhi, **self.preprocessing_kauh}
        return self.data