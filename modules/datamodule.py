"""
Lung Sound Classification DataModule using PyTorch Lightning.
"""
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from typing import Any

from modules.dataset import KAUHAudioDataset, ICBHIAudioDataset, CombinedAudioDataset, LungSoundAudioDataset, DIAGNOSIS
import modules.transforms as T

class LungSoundDataModule(L.LightningDataModule):
    """ PyTorch Lightning DataModule for lung sound datasets. """

    def __init__(self, config: dict, data_path: str) -> None:
        """
        Initialize the DataModule.
        Args:
            config (dict): Configuration dictionary. Expected keys include:
            ```
            - dataset: {
                - name (str): Name of the dataset to use (e.g. "ICBHI", "KAUH", "combined").
                - classes (list[str]): List of class names to include in the dataset. If not provided, defaults to all classes.
                - num_channels (int): Number of channels of the features.
            }
            - batch_size (int): Batch size for the dataloaders.
            - num_workers (int): Number of worker processes for data loading.
            - sampler (str): Type of sampler to use for the training dataloader (e.g. "equalizer"). If None, no sampler is used.
            - seed (int): Random seed for reproducibility.
            - transforms (dict): Dictionary of transformations for each split (train, val, test). Each value can be either a Compose object or a list of transformations.
            ```
            data_path (str): Path to the data directory.
        """
        super().__init__()
        self.AVAILABLE_DATASETS = {
            "ICBHI": ICBHIAudioDataset,
            "KAUH": KAUHAudioDataset,
            "combined": CombinedAudioDataset,
        }

        # Load dataset configurations
        self.config = config
        dataset_config = config.get("dataset", {})
        dataset_name = dataset_config.get("name")
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(f"Invalid dataset: {dataset_name}. Choose from: {tuple(self.AVAILABLE_DATASETS)}")

        self.data_path = data_path
        self.dataset_class: type[LungSoundAudioDataset] = self.AVAILABLE_DATASETS[dataset_name]
        self.classes = dataset_config.get("classes", DIAGNOSIS)
        self.num_channels = dataset_config.get("num_channels", 1)

        # General configurations
        self.batch_size = config.get("batch_size", 16)
        self.num_workers = config.get("num_workers", 2)
        self.sampler = config.get("sampler", None)
        self.seed = config.get("seed", 42)

        # Load transformations
        self.transforms = self.get_transforms(config)


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
            transform=self.transforms,
            classes=self.classes
        )


    @staticmethod
    def get_transforms(config: dict) -> Any:
        """ Load the transformations from the configuration. """
        transformations = config.get("transforms", {})
        if isinstance(transformations, T.Compose):
            return transformations
        if isinstance(transformations, list):
            operations = []
            for transform in config["transforms"]:
                for name, params in transform.items():
                    transform_class = getattr(T, name, None)
                    if transform_class is None:
                        raise ValueError(f"Transform '{name}' not found in modules.transforms.")
                    operations.append(transform_class(**params))
            return T.Compose(operations)
        raise ValueError("Transforms must be either a Compose object or a list of transformations.")


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
                # Convert features to tensor if they are not already
                sample.features = torch.tensor(sample.features, dtype=torch.float32)
            if sample.features.ndim == 2:
                # Add channel dimension for 2D features (e.g. spectrograms)
                sample.features = sample.features.unsqueeze(0)
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
                # Convert features to tensor if they are not already
                sample.features = torch.tensor(sample.features, dtype=torch.float32)
            if sample.features.ndim == 2:
                # Add channel dimension for 2D features (e.g. spectrograms)
                sample.features = sample.features.unsqueeze(0)
            batch_samples.append(sample.features)
            batch_labels.append(label)
            batch_paths.append(str(sample.wav_file))
        return torch.stack(batch_samples), torch.tensor(batch_labels, dtype=torch.long), batch_paths