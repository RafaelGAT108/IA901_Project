# Classifier module using PyTorch Lightning.
import torch
import pytorch_lightning as L
from torch import nn, optim
from sklearn.metrics import classification_report

from modules.dataset import DIAGNOSIS
from modules.model import load_model


class Classifier(L.LightningModule):
    """PyTorch Lightning module for lung sound classification."""
    def __init__(self, config: dict) -> None:
        """
        Initialize model and other parameters.
        Args:
            config (dict): Configuration dictionary.
        """
        super().__init__()
        self.save_hyperparameters(config)

        # Load model
        self.classes = list(DIAGNOSIS.keys())
        num_classes = len(self.classes)
        model_name = self.hparams.model["name"]
        model_architecture = self.hparams.model["architecture"]
        weights = self.hparams.model.get("weights", None)
        freeze_layers = self.hparams.model.get("freeze_layers", False)

        self.model = load_model(model_architecture, model_name, num_classes, weights, freeze_layers)
        self.is_transformer = "transformer" in model_architecture

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass. """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if self.is_transformer:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)
            return self.model(pixel_values=x).logits
        else:
            return self.model(x)

    def configure_optimizers(self) -> tuple[list, list] | optim.Optimizer:
        """ Configure optimizer and learning rate scheduler. """
        if self.hparams.model["freeze_layers"]:
            params = self.model.fc.parameters()
        else:
            params = self.model.parameters()

        opt = self.hparams.optimizer
        sch = self.hparams.lr_scheduler
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        step = self.hparams.step_size
        gamma = self.hparams.gamma
        mtm = self.hparams.momentum

        if opt == "AdamW":
            optimizer = optim.AdamW(params, lr=lr, weight_decay=wd)
        elif opt == "Adam":
            optimizer = optim.Adam(params, lr=lr, weight_decay=wd)
        elif opt == "SGD":
            optimizer = optim.SGD(params, lr=lr, momentum=mtm)
        else:
            raise ValueError(f"Invalid optimizer: {opt}.")

        if sch == "StepLR":
            lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)
        elif sch == None:
            return optimizer
        else:
            raise ValueError(f"Invalid learning rate scheduler: {sch}.")

        return [optimizer], [lr_scheduler]

    def on_train_epoch_start(self) -> None:
        """ Called at the start of the training epoch. """
        self.skipped_on_train = 0

    def training_step(self, batch, batch_idx):
        """ Training step for Pytorch Lightning. """
        if batch is None:
            self.skipped_on_train += 1
            return None
        inputs, labels = batch
        labels = labels.to(dtype=torch.long)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        """ Called at the end of the training epoch. """
        self.log("skipped_on_train", self.skipped_on_train, on_step=False, on_epoch=True, prog_bar=False)

    def on_validation_epoch_start(self) -> None:
        """ Called at the start of the validation epoch. """
        self.skipped_on_val = 0
        self.validation_results = {"preds": [], "labels": []}

    def validation_step(self, batch, batch_idx):
        """ Validation step for Pytorch Lightning. """
        if batch is None:
            self.skipped_on_val += 1
            return None
        inputs, labels = batch
        labels = labels.to(dtype=torch.long)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Store results
        preds = torch.argmax(outputs, dim=1)
        self.validation_results["preds"].extend(preds.detach().cpu().tolist())
        self.validation_results["labels"].extend(labels.detach().cpu().tolist())
        return loss

    def on_validation_epoch_end(self):
        """ Called at the end of the validation epoch. """
        preds = self.validation_results["preds"]
        labels = self.validation_results["labels"]

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
        metrics["val_macro_f1"] = report["macro avg"]["f1-score"]
        metrics["val_weighted_f1"] = report["weighted avg"]["f1-score"]
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)

        self.log("skipped_on_val", self.skipped_on_val, on_step=False, on_epoch=True, prog_bar=False)

    def on_test_epoch_start(self) -> None:
        """ Called at the start of the test epoch. """
        self.skipped_on_test = 0
        self.test_results = {"probs": [], "preds": [], "labels": [], "metadata": []}

    def test_step(self, batch, batch_idx):
        """ Test step for Pytorch Lightning. """
        if batch is None:
            self.skipped_on_test += 1
            return None
        inputs, labels, metadata = batch
        labels = labels.to(dtype=torch.long)
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store results
        self.test_results["preds"].extend(preds.detach().cpu().tolist())
        self.test_results["labels"].extend(labels.detach().cpu().tolist())
        self.test_results["metadata"].extend(metadata)
        return loss

    def on_test_epoch_end(self):
        """ Called at the end of the test epoch. """
        preds = self.test_results["preds"]
        labels = self.test_results["labels"]

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
            metrics[f"test_{class_name}_precision"] = report[class_name]["precision"]
            metrics[f"test_{class_name}_recall"] = report[class_name]["recall"]
            metrics[f"test_{class_name}_f1"] = report[class_name]["f1-score"]
        metrics["test_macro_f1"] = report["macro avg"]["f1-score"]
        metrics["test_weighted_f1"] = report["weighted avg"]["f1-score"]
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)

        self.log("skipped_on_test", self.skipped_on_test, on_step=False, on_epoch=True, prog_bar=False)