import os
import re
from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

from dataset.audio import FraiwanLungSound, ICBHILungSound, LungSound
from dataset.labels import DIAGNOSIS


class LungSoundDataset(Dataset):
    """Base dataset for lung sound collections."""

    def __init__(
            self,
            root,
            transform=None,
            standardize: bool = False,
            target_sr: int = None,
            target_duration: float = None
        ):
        self.root = Path(root)
        self.transform = transform
        if standardize:
            if target_sr is None or target_duration is None:
                raise ValueError("Must specify target_sr and target_duration when standardize=True")
        self.standardize = standardize
        self.target_sr = target_sr
        self.target_duration = target_duration
        self._load_data()
        self._format_data()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        sample = LungSound(row["FilePath"])
        label = row["Label"]

        if self.standardize:
            sample.standardize(self.target_sr, self.target_duration)
        if self.transform is not None:
            sample.apply_transform(self.transform)

        return sample, label

    def _load_data(self) -> pd.DataFrame:
        raise NotImplementedError

    def _format_data(self):
        """
        Format the loaded data into a pandas DataFrame.
        """
        # 1. Filter labels to only include those in the DIAGNOSIS list
        self.data = self.data[self.data["Label"].isin(DIAGNOSIS)]
        # 2. Reset index after filtering
        self.data.reset_index(drop=True, inplace=True)

    def _find_wav_files(self):
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
        sample.plot_audio(title=f"Sample {idx} - Label: {label}")


class ICBHIDataset(LungSoundDataset):
    """Dataset wrapper for the ICBHI respiratory sound database."""

    def __init__(
            self,
            root,
            transform=None,
            standardize: bool = False,
            target_sr: int = None,
            target_duration: float = None
        ):
        data_dir = os.path.join(root, "ICBHI_final_database")
        super().__init__(data_dir, transform, standardize, target_sr, target_duration)

    def _load_data(self) -> pd.DataFrame:
        diagnosis_file = os.path.join(self.root, "ICBHI_Challenge_diagnosis.txt")
        diagnosis_df = pd.read_csv(
            diagnosis_file,
            names=["PatientNumber", "Diagnosis"],
            delimiter="\t",
            dtype={"PatientNumber": str, "Diagnosis": str},
        )
        diagnosis_map = dict(zip(diagnosis_df["PatientNumber"], diagnosis_df["Diagnosis"]))

        records = []
        for wav_file in self._find_wav_files():
            metadata = ICBHILungSound.parse_metadata(wav_file)
            records.append(
                {
                    "FilePath": str(wav_file),
                    "Label": diagnosis_map.get(metadata["PatientNumber"]),
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)


class FraiwanDataset(LungSoundDataset):
    """Dataset wrapper for the Fraiwan et al. respiratory sound database."""
    def __init__(
            self,
            root,
            transform=None,
            standardize: bool = False,
            target_sr: int = None,
            target_duration: float = None
        ):
        data_dir = os.path.join(root, "fraiwan", "Audio Files")
        super().__init__(data_dir, transform, standardize, target_sr, target_duration)

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _canonical_diagnosis(raw_label: str | None) -> str | None:
        if raw_label is None:
            return None

        label = FraiwanDataset._normalize_text(raw_label).lower()
        label = label.replace(".", "")

        canonical_map = {
            "n": "Healthy",
            "normal": "Healthy",
            "healthy": "Healthy",
            "asthma": "Asthma",
            "copd": "COPD",
            "pneumonia": "Pneumonia",
            "bronchiolitis": "Bronchiolitis",
            "bronchiectasis": "Bronchiectasis",
            "urti": "URTI",
            "lrti": "LRTI",
        }

        return canonical_map.get(label)

    @staticmethod
    def _parse_filename_annotation(stem: str) -> dict:
        # Example: BP101_Asthma,E W,P L M,12,F
        if "_" not in stem:
            return {}

        _, metadata_part = stem.split("_", 1)
        parts = [part.strip() for part in metadata_part.split(",", 4)]
        if len(parts) != 5:
            return {}

        diagnosis, sound_type, location, age, gender = parts
        return {
            "Diagnosis": diagnosis,
            "Sound type": sound_type,
            "Location": location,
            "Age": age,
            "Gender": gender,
        }

    def _build_fraiwan_annotation_map(self, annotation_df: pd.DataFrame) -> dict:
        expected_columns = ["Age", "Gender", "Location", "Sound type", "Diagnosis"]
        missing = [column for column in expected_columns if column not in annotation_df.columns]
        if missing:
            raise ValueError(f"Missing required columns in Fraiwan annotation file: {missing}")

        annotation_map = {}
        for _, row in annotation_df.iterrows():
            key = (
                self._normalize_text(row["Age"]),
                self._normalize_text(row["Gender"]).upper(),
                self._normalize_text(row["Location"]),
                self._normalize_text(row["Sound type"]),
            )
            annotation_map[key] = self._normalize_text(row["Diagnosis"])

        return annotation_map

    def _load_data(self) -> pd.DataFrame:
        annotation_file = Path(self.root).parent / "Data annotation.xlsx"
        try:
            annotation_df = pd.read_excel(annotation_file)
        except ImportError as exc:
            raise ImportError(
                "openpyxl is required to read Fraiwan annotation file (.xlsx). "
                "Install it with `pip install openpyxl` or `conda install openpyxl`."
            ) from exc

        annotation_map = self._build_fraiwan_annotation_map(annotation_df)

        records = []
        for wav_file in self._find_wav_files():
            metadata = FraiwanLungSound.parse_metadata(wav_file)

            filename_fields = self._parse_filename_annotation(Path(wav_file).stem)
            lookup_key = (
                self._normalize_text(filename_fields.get("Age")),
                self._normalize_text(filename_fields.get("Gender")).upper(),
                self._normalize_text(filename_fields.get("Location")),
                self._normalize_text(filename_fields.get("Sound type")),
            )

            raw_diagnosis = annotation_map.get(lookup_key, filename_fields.get("Diagnosis"))
            label = self._canonical_diagnosis(raw_diagnosis)

            records.append(
                {
                    "FilePath": str(wav_file),
                    "Label": label,
                    "RawDiagnosis": raw_diagnosis,
                    **metadata,
                }
            )
        self.data = pd.DataFrame(records)