import os
from typing import Any

import numpy as np
import lightning as L
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler

from modules.dataset import FraiwanDataset, ICBHIDataset, CombinedLungSoundDataset, LungSoundDataset, DIAGNOSIS
import modules.transforms as T

class LungSoundDataModule(L.LightningDataModule):
    """PyTorch Lightning DataModule for lung sound datasets."""

    def __init__(self, config: dict, data_path: str) -> None:
        """
        Initialize the DataModule.
        Args:
            config (dict): Configuration dictionary.
            data_path (str): Path to the data directory.
        """
        super().__init__()
        self.AVAILABLE_DATASETS = {
            "icbhi": ICBHIDataset,
            "fraiwan": FraiwanDataset,
            "combined": CombinedLungSoundDataset,
        }

        # Load dataset configurations
        dataset_config = config.get("dataset", {})
        dataset_name = dataset_config.get("name")
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Invalid dataset: {dataset_name}. Choose from: {tuple(self.AVAILABLE_DATASETS)}")

        self.data_path = data_path
        self.dataset_class: type[LungSoundDataset] = self.AVAILABLE_DATASETS[dataset_name]
        self.classes = dataset_config.get("classes", DIAGNOSIS)

        # General configurations
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", 2)
        self.sampler = config.get("sampler", None)
        self.seed = config.get("seed", 42)

        # Load transformations
        self.transforms = {
            "train": self.get_transforms(config, "train"),
            "val": self.get_transforms(config, "val"),
            "test": self.get_transforms(config, "test"),
        }


    def setup(self, stage: str | None = None) -> None:
        """ Set up the datasets for training, validation, and testing. """
        if stage in (None, "fit"):
            self.train_dataset = self.create_dataset("train")
            self.val_dataset = self.create_dataset("val")

        if stage in (None, "test"):
            self.test_dataset = self.create_dataset("test")


    def create_dataset(self, split: str) -> Dataset:
        return self.dataset_class(
            root=self.data_path,
            split=split,
            transform=self.transforms[split],
            classes=self.classes
        )


    @staticmethod
    def get_transforms(config: dict, split: str) -> Any:
        """ Load the transformations for a given split based on the configuration. """
        transformations = config.get("transforms", {})
        if not isinstance(transformations, dict):
            raise ValueError("Transforms configuration must be a dictionary with splits as keys.")
        if split not in transformations:
            raise ValueError(f"Transforms for split '{split}' not found in configuration.")
        if isinstance(transformations[split], T.Compose):
            return transformations[split]
        if isinstance(transformations[split], list):
            try:
                compose = T.Compose(transformations[split])
                return compose
            except Exception as e:
                print(f"Error creating transforms for split '{split}': {e}")
                pass
        operations = []
        for transform in config["transforms"][split]:
            for name, params in transform.items():
                transform_class = getattr(T, name, None)
                if transform_class is None:
                    raise ValueError(f"Transform '{name}' not found in modules.transforms.")
                operations.append(transform_class(**params))
        return T.Compose(operations)


    @staticmethod
    def get_equalizer_sampler(dataset: Dataset) -> WeightedRandomSampler:
        """
        Function to get the sampler for the dataset.
        Adapted from: https://apxml.com/courses/getting-started-with-pytorch/chapter-5-efficient-data-handling/customizing-dataloader-behavior
        """
        # Getting all the labels from the dataset as a list
        # labels = [dataset[i][1] for i in range(len(dataset))]
        labels = dataset.data["Label"].tolist()
        # Calculating the count of each class in the dataset
        class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
        # Calculating the weights for each sample based on the inverse of the class counts
        # Oversampling strategy: it gives higher weight to samples from underrepresented classes
        sample_weights = torch.tensor([1.0 / class_counts[l] for l in labels])
        num_samples = len(labels)
        return WeightedRandomSampler(weights=sample_weights, num_samples=num_samples)


    def train_dataloader(self) -> DataLoader:
        """ Return the training dataloader. """
        sampler = None
        if self.sampler == "equalizer":
            sampler = self.get_equalizer_sampler(self.train_dataset)
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=self.custom_collate_fn,
        )


    def val_dataloader(self) -> DataLoader:
        """ Return the validation dataloader. """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
        )


    def test_dataloader(self) -> DataLoader:
        """ Return the test dataloader. """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            shuffle=False,
            collate_fn=self.test_collate_fn,
        )


    @staticmethod
    def custom_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
        """ Custom collate function to handle variable-length audio samples. """
        batch_samples = []
        batch_labels = []
        for sample, label in batch:
            if not isinstance(sample.features, torch.Tensor):
                sample.features = torch.tensor(sample.features, dtype=torch.float32)
            batch_samples.append(sample.features)
            batch_labels.append(label)
        return torch.stack(batch_samples), torch.tensor(batch_labels, dtype=torch.long)


    @staticmethod
    def test_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        """ Custom collate function for the test set to also return file paths. """
        batch_samples = []
        batch_labels = []
        batch_paths = []
        for sample, label in batch:
            if not isinstance(sample.features, torch.Tensor):
                sample.features = torch.tensor(sample.features, dtype=torch.float32)
            batch_samples.append(sample.features)
            batch_labels.append(label)
            batch_paths.append(sample.wav_file)
        return torch.stack(batch_samples), torch.tensor(batch_labels, dtype=torch.long), batch_paths