import os
import re
from pathlib import Path
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from modules.audio import LungSound
from modules.preprocess import *

DIAGNOSIS_ICBHI = [
    "Asthma",
    "Bronchiectasis",
    "Bronchiolitis",
    "COPD",             # Chronic Obstructive Pulmonary Disease
    "Healthy",          # Healthy/Normal
    "LRTI",             # Lower Respiratory Tract Infection
    "URTI",             # Upper Respiratory Tract Infection
    "Pneumonia",
]

DIAGNOSIS_FRAIWAN = [
    "Asthma",
    "Bronchitis",       # Bronchitis
    "COPD",
    "Hearth Failure",
    "Lung Fibrosis",
    "Normal",           # Healthy/Normal
    "Pleural Effusion",
    "Pneumonia",
]

DIAGNOSIS = {
    "Asthma": ["asthma"],
    "Bronchiectasis": ["bronchiectasis"],
    "Bronchiolitis": ["bronchiolitis"],
    "COPD": ["copd"],
    "Healthy": ["healthy","n"],
    "Lung Fibrosis": ["lung fibrosis"],
    "Pneumonia": ["pneumonia"],
    "URTI": ["urti"],
    # NOTE: Removed due to very low sample count:
    # "Pleural Effusion": ["pleural effusion", "plueral effusion"],
    # "Bronchitis": ["bron"],
    # "Hearth Failure": ["heart failure"],
    # "LRTI": ["lrti"],
}

# Create a mapping from diagnosis values (lowercase) to actual diagnosis names
DIAGNOSIS_MAP = {}
for key, values in DIAGNOSIS.items():
    for value in values:
        DIAGNOSIS_MAP[value] = key


class LungSoundDataset(Dataset):
    """
    Base dataset for lung sound collections.
    """
    def __init__(
            self,
            root: str | Path,
            split: str,
            preprocess: AudioTransform | FeatureExtractor | Compose | None = None,
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all".
            preprocess (AudioTransform | FeatureExtractor | Compose | None): Optional preprocessing pipeline to apply to each sample.
        """
        self.root = Path(root)
        self.split = split
        self.preprocess = preprocess
        self.classes = list(DIAGNOSIS.keys())
        self._load_data()
        self._filter_classes()
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

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        return sample, label

    def _load_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _handle_windowing(self) -> None:
        """
        Handle windowing to the dataset by expanding the DataFrame with new rows for each window.
        """
        self.windowing = False
        if self.preprocess is None:
            return
        if isinstance(self.preprocess, Compose):
            # Check if there is a Window transform in the preprocessing pipeline
            for transform in self.preprocess.transforms:
                if isinstance(transform, Window):
                    self.data = transform.expand_dataframe(self.data)
                    self.windowing = True
                    return
        if isinstance(self.preprocess, Window):
            self.data = self.preprocess.expand_dataframe(self.data)
            self.windowing = True
            return
        # No windowing transform found in the preprocessing pipeline

    def _filter_classes(self):
        """
        Filter the dataset to only include samples with labels in our DIAGNOSIS mapping.
        """
        # 1. Convert labels to lowercase and map to canonical diagnosis names
        self.data["Label"] = self.data["Label"].str.lower().map(DIAGNOSIS_MAP)
        # 2. Filter out any labels that weren't found in the mapping (NaN values)
        self.data = self.data.dropna(subset=["Label"])
        # 3. Remove classes that have very low sample counts
        # class_counts = self.data["Label"].value_counts()
        # valid_labels = []
        # for label, label_count in class_counts.items():
        #     if label_count < 9:
        #         print(f"Warning: Class '{label}' has only {label_count} samples. it will be removed from the dataset.")
        #     else:
        #         valid_labels.append(label)
        # self.data = self.data[self.data["Label"].isin(valid_labels)]
        self.classes = sorted(self.data["Label"].unique())
        # 4. Reset index after filtering
        self.data.reset_index(drop=True, inplace=True)

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
        print(f"Sample {idx}:")
        print(f"  File path: {sample.wav_file}")
        print(f"  Sample rate: {sample.sr}")
        print(f"  Audio shape: {sample.audio.shape}")
        print(f"  Label: {label}")
        # Plot the waveform
        sample.plot_waveform(title=f"Sample {idx} - Label: {label}")
        # Plot the features if they exist
        if sample.has_features:
            sample.plot_features(title=f"Sample {idx} - Features")


class ICBHIDataset(LungSoundDataset):
    """ Dataset for the ICBHI Challenge (2017) respiratory sound database."""
    def __init__(
            self,
            root: str | Path,
            split: str,
            preprocess: AudioTransform | FeatureExtractor | Compose | None = None,
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all".
            preprocess (AudioTransform | FeatureExtractor | Compose | None): Optional preprocessing pipeline to apply to each sample.
        """
        data_dir = os.path.join(root, "ICBHI_final_database")
        super().__init__(data_dir, split, preprocess)

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
                    "FilePath": str(wav_file),
                    "Label": diagnosis_map.get(patient_number, "Unknown"),
                    **demographic_info,
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)


class FraiwanDataset(LungSoundDataset):
    """ Dataset for the Fraiwan et al. (2021) respiratory sound database. """
    def __init__(
            self,
            root: str | Path,
            split: str,
            preprocess: AudioTransform | FeatureExtractor | Compose | None = None,
        ):
        """
        Initialize the dataset.
        Args:
            root (str | Path): Root directory of the dataset.
            split (str): Dataset split. Can be "train", "val", "test", or "all".
            preprocess (AudioTransform | FeatureExtractor | Compose | None): Optional preprocessing pipeline to apply to each sample.
        """
        data_dir = os.path.join(root, "fraiwan")
        super().__init__(data_dir, split, preprocess)

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

        return {
            "Label": diagnosis,
            "FilterCode": filter_code,
            "PatientId": patient_num,
            "Sound type": sound_type,
            "Location": location,
            "Age": float(age),
            "Sex": sex,
        }

    def _load_data(self) -> pd.DataFrame:
        """ Load data from the Fraiwan dataset. """
        records = []
        for wav_file in self._find_wav_files():
            metadata = self.parse_metadata(wav_file)
            records.append(
                {
                    "FilePath": str(wav_file),
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)