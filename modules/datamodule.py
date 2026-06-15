"""
Lung Sound Classification DataModule using PyTorch Lightning.
"""

import numpy as np
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from typing import Any

from modules.datasets import KAUHFeaturesDataset, ICBHIFeaturesDataset, CombinedFeaturesDataset, FeaturesDataset, DIAGNOSIS

class LungSoundDataModule(L.LightningDataModule):
    """ PyTorch Lightning DataModule for lung sound datasets. """

    def __init__(self, config: dict, data_path: str) -> None:
        """
        Initialize the DataModule.
        Args:
            config (dict): Configuration dictionary. Expected keys include:
            ```
            - dataset: {
                - name (str): Name of the dataset to use (e.g. "ICBHI", "KAUH" or "Combined_ICBHI_KAUH").
                - classes (list[str]): List of class names to include in the dataset. If not provided, defaults to all classes.
                - feature_extractor (str | list[str]): Name or list of names of the feature extractors used during preprocessing. This is used to determine how to load the features in the dataset class.
                - sample_limit (int | None): Maximum number of samples per class to include in the dataset. If None, include all samples.
                - transforms (FeatureTransform | Compose | None): Optional feature transformations to apply to the features when loading the dataset.
            }
            - batch_size (int): Batch size for the dataloaders.
            - num_workers (int): Number of worker processes for data loading.
            - sampler (str): Type of sampler to use for the training dataloader (e.g. "equalizer"). If None, no sampler is used.
            - seed (int): Random seed for reproducibility.
            ```
            data_path (str): Path to the data directory.
        """
        super().__init__()
        self.AVAILABLE_DATASETS = {
            "ICBHI": ICBHIFeaturesDataset,
            "KAUH": KAUHFeaturesDataset,
            "Combined_ICBHI_KAUH": CombinedFeaturesDataset,
        }

        # Load dataset configurations
        self.config = config
        dataset_config = config.get("dataset", {})
        dataset_name = dataset_config.get("name")
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Invalid dataset: {dataset_name}. Choose from: {tuple(self.AVAILABLE_DATASETS)}")

        self.data_path = data_path
        self.dataset_class: type[FeaturesDataset] = self.AVAILABLE_DATASETS[dataset_name]
        self.classes = dataset_config.get("classes", DIAGNOSIS)
        self.transforms = dataset_config.get("transforms", None)
        self.feature_extractor = dataset_config.get("feature_extractor", "MagSTFT")
        self.num_channels = len(self.feature_extractor) if isinstance(self.feature_extractor, list) else 1
        self.sample_limit = dataset_config.get("sample_limit", None)

        # General configurations
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", 2)
        self.sampler = config.get("sampler", None)
        self.seed = config.get("seed", 42)


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
            feature_extractor=self.feature_extractor,
            split=split,
            classes=self.classes,
            transform=self.transforms,
            random_seed=self.seed,
            sample_limit=self.sample_limit,
        )


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
    def convert_to_tensor(sample: Any) -> torch.Tensor:
        """ Utility function to convert a sample to a PyTorch tensor. """
        if isinstance(sample, torch.Tensor):
            return sample.to(torch.float32)
        elif isinstance(sample, np.ndarray):
            tensor = torch.tensor(sample, dtype=torch.float32)
            if tensor.ndim == 2:
                # Add channel dimension for 2D features (e.g. spectrograms)
                return tensor.unsqueeze(0)
            elif tensor.ndim == 3:
                # Permute dimensions to (channels, height, width) for 3D features (e.g. stacked features)
                return tensor.permute(2, 0, 1)
            else:
                raise ValueError(f"Unsupported feature shape: {tensor.shape}.")
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}. Expected np.ndarray or torch.Tensor.")

    @staticmethod
    def custom_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """ Custom collate function to handle variable-length audio samples. """
        batch_features = []
        batch_labels = []
        batch_sr = []
        for sample, label in batch:
            batch_features.append(LungSoundDataModule.convert_to_tensor(sample.features))
            batch_labels.append(label)
            batch_sr.append(sample.sr)
        return torch.stack(batch_features), torch.tensor(batch_labels, dtype=torch.long), batch_sr


    @staticmethod
    def test_collate_fn(batch: list) -> tuple[torch.Tensor, torch.Tensor, list[int], list[str]]:
        """ Custom collate function for the test set to also return file paths. """
        batch_features = []
        batch_labels = []
        batch_sr = []
        batch_paths = []
        for sample, label in batch:
            batch_features.append(LungSoundDataModule.convert_to_tensor(sample.features))
            batch_labels.append(label)
            batch_sr.append(sample.sr)
            batch_paths.append(sample.info)
        return torch.stack(batch_features), torch.tensor(batch_labels, dtype=torch.long), batch_sr, batch_paths