"""
Lung Sound Classification Module using PyTorch Lightning.
"""

import torch
import lightning as L
from torch import nn, optim
from torchmetrics.classification import F1Score, Accuracy
from sklearn.metrics import classification_report

from modules.model import LungSoundModel


class LungSoundClassifier(L.LightningModule):
    """ PyTorch Lightning module for lung sound classification. """
    def __init__(self, config: dict) -> None:
        """
        Initialize model and other parameters.
        Args:
            config (dict): Configuration dictionary. Expected keys include:
            ```
            - model: {
                - name (str): Name of the model to load (e.g. "resnet18").
                - pretrained (str, optional): Pre-trained weights to load. Defaults to None.
                - freeze_layers (bool): Whether to freeze the layers of the pre-trained model. Defaults to False.
            }
            - dataset: {
                - classes (list[str]): List of class names in the dataset.
                - num_channels (int): Number of channels in the input data.
            }
            - criterion (str): Loss function to use (e.g. "CrossEntropyLoss").
            - optimizer (str): Optimizer to use (e.g. "AdamW").
            - learning_rate (float): Learning rate for the optimizer.
            - weight_decay (float): Weight decay for the optimizer (if applicable).
            - momentum (float): Momentum for the optimizer (if applicable).
            - lr_scheduler (str, optional): Learning rate scheduler to use (e.g. "StepLR"). If None, no scheduler is used. Defaults to None.
            - step_size (int): Step size for the learning rate scheduler (if applicable).
            - gamma (float): Gamma for the learning rate scheduler (if applicable).
            ```
        """
        super().__init__()
        self.save_hyperparameters(config)

        # Load model
        self.classes = self.hparams.dataset["classes"]
        self.lung_sound_model = LungSoundModel(
            name=self.hparams.model["name"],
            num_channels=self.hparams.dataset["num_channels"],
            num_classes=len(self.classes),
            pretrained=self.hparams.model.get("pretrained", None),
            freeze_layers=self.hparams.model.get("freeze_layers", False)
        )
        self.model = self.lung_sound_model.model
        self.trainable_params = self.lung_sound_model.trainable_params

        # Configure loss function
        self.configure_loss()

        # Configure metrics
        self.train_f1_macro = F1Score(task="multiclass", num_classes=len(self.classes), average="macro")
        self.train_f1_micro = F1Score(task="multiclass", num_classes=len(self.classes), average="micro")
        self.train_acc = Accuracy(task="multiclass", num_classes=len(self.classes))
        self.val_f1_macro = F1Score(task="multiclass", num_classes=len(self.classes), average="macro")
        self.val_f1_micro = F1Score(task="multiclass", num_classes=len(self.classes), average="micro")
        self.val_acc = Accuracy(task="multiclass", num_classes=len(self.classes))
        self.test_f1_macro = F1Score(task="multiclass", num_classes=len(self.classes), average="macro")
        self.test_f1_micro = F1Score(task="multiclass", num_classes=len(self.classes), average="micro")
        self.test_acc = Accuracy(task="multiclass", num_classes=len(self.classes))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        return self.model(x)


    def configure_optimizers(self) -> tuple[list, list] | optim.Optimizer:
        """ Configure optimizer and learning rate scheduler. """
        opt = self.hparams.optimizer
        sch = self.hparams.lr_scheduler
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        mtm = self.hparams.momentum
        step = self.hparams.step_size
        gamma = self.hparams.gamma

        if opt == "AdamW":
            optimizer = optim.AdamW(self.trainable_params, lr=lr, weight_decay=wd)
        elif opt == "Adam":
            optimizer = optim.Adam(self.trainable_params, lr=lr, weight_decay=wd)
        elif opt == "SGD":
            optimizer = optim.SGD(self.trainable_params, lr=lr, momentum=mtm)
        else:
            raise ValueError(f"Invalid optimizer: {opt}.")

        if sch == "StepLR":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
        elif sch == None:
            return optimizer
        else:
            raise ValueError(f"Invalid learning rate scheduler: {sch}.")

        return [optimizer], [lr_scheduler]


    def configure_loss(self):
        """ Configure loss function. """
        if self.hparams.criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Invalid loss function: {self.hparams.criterion}.")


    def training_step(self, batch, batch_idx):
        """ Training step for Pytorch Lightning. """
        inputs, labels, sr = batch
        labels = labels.to(dtype=torch.long)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
    
        # Log metrics
        self.train_f1_macro(outputs, labels)
        self.train_f1_micro(outputs, labels)
        self.train_acc(outputs, labels)
        self.log("train_f1_macro", self.train_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_f1_micro", self.train_f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss


    def on_validation_epoch_start(self) -> None:
        """ Called at the start of the validation epoch. """
        self.validation_results = {"preds": [], "targets": []}


    def validation_step(self, batch, batch_idx):
        """ Validation step for Pytorch Lightning. """
        inputs, labels, sr = batch
        labels = labels.to(dtype=torch.long)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        # Log metrics
        self.val_f1_macro(outputs, labels)
        self.val_f1_micro(outputs, labels)
        self.val_acc(outputs, labels)
        self.log("val_f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1_micro", self.val_f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store results
        preds = torch.argmax(outputs, dim=1)
        self.validation_results["preds"].extend(preds.detach().cpu().tolist())
        self.validation_results["targets"].extend(labels.detach().cpu().tolist())
        return loss


    def on_validation_epoch_end(self):
        """ Called at the end of the validation epoch. """
        preds = self.validation_results["preds"]
        labels = self.validation_results["targets"]

        # Generate classification report
        report = classification_report(
            labels,
            preds,
            output_dict=True,
            target_names=self.classes,
            labels=list(range(len(self.classes))),
            zero_division=0
        )

        # Log classification report
        metrics = {}
        for class_name in self.classes:
            metrics[f"val_{class_name}_f1"] = report[class_name]["f1-score"]
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)


    def on_test_epoch_start(self) -> None:
        """ Called at the start of the test epoch. """
        self.test_results = {"probs": [], "preds": [], "targets": [], "info": []}


    def test_step(self, batch, batch_idx):
        """ Test step for Pytorch Lightning. """
        inputs, labels, sr, info = batch
        labels = labels.to(dtype=torch.long)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log metrics
        self.test_f1_macro(outputs, labels)
        self.test_f1_micro(outputs, labels)
        self.test_acc(outputs, labels)
        self.log("test_f1_macro", self.test_f1_macro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_f1_micro", self.test_f1_micro, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store results
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, dim=1)
        self.test_results["probs"].extend(probs.detach().cpu().tolist())
        self.test_results["preds"].extend(preds.detach().cpu().tolist())
        self.test_results["targets"].extend(labels.detach().cpu().tolist())
        self.test_results["info"].extend(info)
        return loss


    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. """
        preds = self.test_results["preds"]
        targets = self.test_results["targets"]

        # Generate classification report
        report = classification_report(
            y_true=targets,
            y_pred=preds,
            output_dict=True,
            target_names=self.classes,
            labels=list(range(len(self.classes))),
            zero_division=0
        )

        # Log classification report
        metrics = {}
        for class_name in self.classes:
            metrics[f"test_{class_name}_precision"] = report[class_name]["precision"]
            metrics[f"test_{class_name}_recall"] = report[class_name]["recall"]
            metrics[f"test_{class_name}_f1"] = report[class_name]["f1-score"]
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)