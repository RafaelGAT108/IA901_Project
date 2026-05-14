import os
from typing import Any

import numpy as np
try:
    import pytorch_lightning as L
except ModuleNotFoundError:
    class _LightningDataModule:
        pass

    class _LightningNamespace:
        LightningDataModule = _LightningDataModule

    L = _LightningNamespace()
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torch.utils.data.sampler import WeightedRandomSampler

from dataset import FraiwanDataset, ICBHIDataset
from dataset.labels import DIAGNOSIS
from dataset.transforms import (
    CQTSpectrogramTransform,
    ChromaTransform,
    MFCCDeltaTransform,
    MFCCTransform,
    MelSpectrogramTransform,
    PhaseTransform,
    RawWaveformTransform,
    SpectralContrastTransform,
    SpectrogramTransform,
)


AVAILABLE_DATASETS = {
    "icbhi": ICBHIDataset,
    "fraiwan": FraiwanDataset,
}

AVAILABLE_TRANSFORMS = {
    "SpectrogramTransform": SpectrogramTransform,
    "MelSpectrogramTransform": MelSpectrogramTransform,
    "MFCCTransform": MFCCTransform,
    "MFCCDeltaTransform": MFCCDeltaTransform,
    "ChromaTransform": ChromaTransform,
    "SpectralContrastTransform": SpectralContrastTransform,
    "CQTSpectrogramTransform": CQTSpectrogramTransform,
    "PhaseTransform": PhaseTransform,
    "RawWaveformTransform": RawWaveformTransform,
}


class DataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for lung sound datasets."""

    def __init__(self, config: dict, data_path: str) -> None:
        super().__init__()
        datasets_config = config.get("datasets", {})
        audio_config = config.get("audio", {})

        self.data_path = data_path
        self.batch_size = config.get("batch_size", 32)
        self.num_workers = config.get("num_workers", 4)
        self.seed = config.get("seed", 42)
        self.oversample = datasets_config.get("oversample", False)

        self.class_names = list(DIAGNOSIS.keys())
        self.label_to_index = {label: idx for idx, label in enumerate(self.class_names)}

        self.datasets = datasets_config.get("names")
        if not self.datasets:
            raise ValueError("No dataset specified.")
        for dataset_name in self.datasets:
            if dataset_name not in AVAILABLE_DATASETS:
                raise ValueError(f"Invalid dataset: {dataset_name}. Choose from: {tuple(AVAILABLE_DATASETS)}")

        self.test_sets = datasets_config.get("test_datasets", self.datasets)
        for dataset_name in self.test_sets:
            if dataset_name not in AVAILABLE_DATASETS:
                raise ValueError(f"Invalid test dataset: {dataset_name}. Choose from: {tuple(AVAILABLE_DATASETS)}")

        self.standardize = audio_config.get("standardize", True)
        self.target_sr = audio_config.get("target_sr", 18000)
        self.target_duration = audio_config.get("target_duration", 20)
        self.train_ratio = datasets_config.get("train_ratio", 0.7)
        self.val_ratio = datasets_config.get("val_ratio", 0.15)
        self.test_ratio = datasets_config.get("test_ratio", 0.15)

        self.transforms = {
            "train": self.get_transform(config, "train"),
            "val": self.get_transform(config, "val"),
            "test": self.get_transform(config, "test"),
        }

    def setup(self, stage: str | None = None) -> None:
        if stage in ("fit", None):
            train_datasets = []
            val_datasets = []
            for dataset_name in self.datasets:
                train_dataset, val_dataset, _ = self._build_splits(dataset_name)
                train_datasets.append(train_dataset)
                val_datasets.append(val_dataset)

            self.train_dataset = ConcatDataset(train_datasets)
            self.val_dataset = ConcatDataset(val_datasets)

            print(f"\nTraining on: {self.datasets}")
            print(f"Number of samples on train set: {len(self.train_dataset)}")
            print(f"Number of samples on valid set: {len(self.val_dataset)}\n")

        if stage in ("test", None):
            test_datasets = []
            for dataset_name in self.test_sets:
                _, _, test_dataset = self._build_splits(dataset_name)
                test_datasets.append(test_dataset)

            self.test_dataset = ConcatDataset(test_datasets)

            print(f"\nEvaluating on: {self.test_sets}")
            print(f"Number of samples on test set: {len(self.test_dataset)}\n")

    def _build_splits(self, dataset_name: str) -> tuple[Subset, Subset, Subset]:
        DatasetClass = AVAILABLE_DATASETS[dataset_name]
        train_dataset = DatasetClass(
            self.data_path,
            transform=self.transforms["train"],
            standardize=self.standardize,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
        )
        val_dataset = DatasetClass(
            self.data_path,
            transform=self.transforms["val"],
            standardize=self.standardize,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
        )
        test_dataset = DatasetClass(
            self.data_path,
            transform=self.transforms["test"],
            standardize=self.standardize,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
        )

        dataset_len = len(train_dataset)
        train_len = int(dataset_len * self.train_ratio)
        val_len = int(dataset_len * self.val_ratio)
        test_len = dataset_len - train_len - val_len
        if test_len < 0:
            raise ValueError("Train/val/test ratios are invalid.")

        seed_offset = sum(ord(char) for char in dataset_name)
        generator = torch.Generator().manual_seed(self.seed + seed_offset)
        indices = torch.randperm(dataset_len, generator=generator).tolist()
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        return (
            Subset(train_dataset, train_indices),
            Subset(val_dataset, val_indices),
            Subset(test_dataset, test_indices),
        )

    def create_dataset(self, dataset_name: str) -> Dataset:
        """Create a dataset with the train transform. Kept for compatibility."""
        DatasetClass = AVAILABLE_DATASETS[dataset_name]
        return DatasetClass(
            self.data_path,
            transform=self.transforms["train"],
            standardize=self.standardize,
            target_sr=self.target_sr,
            target_duration=self.target_duration,
        )

    def train_dataloader(self):
        sampler = self.get_sampler(self.train_dataset) if self.oversample else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.train_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self.test_collate_fn,
        )

    def _collect_labels(self, dataset: Dataset) -> list[str]:
        if isinstance(dataset, ConcatDataset):
            labels: list[str] = []
            for sub_dataset in dataset.datasets:
                labels.extend(self._collect_labels(sub_dataset))
            return labels

        if isinstance(dataset, Subset):
            return dataset.dataset.data.iloc[dataset.indices]["Label"].tolist()

        return dataset.data["Label"].tolist()

    def get_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        labels = self._collect_labels(dataset)
        label_indices = np.array([self.label_to_index[label] for label in labels], dtype=np.int64)
        class_counts = np.bincount(label_indices, minlength=len(self.class_names))
        class_counts[class_counts == 0] = 1
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[label_indices]
        return WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), len(sample_weights), replacement=True)

    def get_transform(self, config: dict, mode: str):
        transform_config = config.get("transforms", {}).get(mode, None)
        if transform_config is None:
            return None

        if isinstance(transform_config, list):
            if len(transform_config) != 1:
                raise ValueError("Use a single audio transform per mode.")
            transform_config = transform_config[0]

        if len(transform_config) != 1:
            raise ValueError("Each audio transform specification must contain a single transform name.")

        transform_name, params = next(iter(transform_config.items()))
        if transform_name not in AVAILABLE_TRANSFORMS:
            raise ValueError(f"Invalid transform: {transform_name}. Choose from: {tuple(AVAILABLE_TRANSFORMS)}")

        transform_class = AVAILABLE_TRANSFORMS[transform_name]
        params = params or {}
        return transform_class(**params)

    def _prepare_sample(self, sample):
        feature = torch.as_tensor(sample.audio, dtype=torch.float32)
        if feature.ndim == 1:
            feature = feature.unsqueeze(0).unsqueeze(0)
        elif feature.ndim == 2:
            feature = feature.unsqueeze(0)
        elif feature.ndim != 3:
            raise ValueError("Audio feature must be a 1D or 2D array.")

        if feature.shape[0] == 1:
            feature = feature.repeat(3, *([1] * (feature.ndim - 1)))
        return feature

    def train_collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor] | None:
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None

        features = []
        labels = []
        for sample, label in batch:
            features.append(self._prepare_sample(sample))
            labels.append(self.label_to_index[label])

        return torch.stack(features), torch.as_tensor(labels, dtype=torch.long)

    def test_collate_fn(self, batch: list) -> tuple[torch.Tensor, torch.Tensor, list[dict]] | None:
        batch = [item for item in batch if item is not None]
        if len(batch) == 0:
            return None, None, None

        features = []
        labels = []
        metadata = []
        for sample, label in batch:
            features.append(self._prepare_sample(sample))
            labels.append(self.label_to_index[label])
            metadata.append({"file_path": str(sample.wav_file), "label": label})

        return torch.stack(features), torch.as_tensor(labels, dtype=torch.long), metadata