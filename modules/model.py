"""
Model loading and architecture definition for lung sound classification.
"""

import torch.nn as nn
import torchvision.models as models

AVAILABLE_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "densenet121": models.densenet121,
    "densenet161": models.densenet161,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    "efficientnet-b0": models.efficientnet_b0,
    "efficientnet-b1": models.efficientnet_b1,
    "efficientnet-b2": models.efficientnet_b2,
    "efficientnet-b3": models.efficientnet_b3,
    "efficientnet-b4": models.efficientnet_b4,
    "efficientnet-b5": models.efficientnet_b5,
    "efficientnet-b6": models.efficientnet_b6,
    "efficientnet-b7": models.efficientnet_b7,
    "efficientnet-v2-s": models.efficientnet_v2_s,
    "efficientnet-v2-m": models.efficientnet_v2_m,
    "efficientnet-v2-l": models.efficientnet_v2_l,
}

# NOTE: Always verify the normalization and preprocessing steps required for each pre-trained model.

def load_model(
        architecture: str,
        name: str,
        num_channels: int,
        num_classes: int,
        pretrained: str = None,
       freeze_layers: bool = False
    ) -> nn.Module:
    """
    Load a model. The output layer is adjusted to match the number of classes.
    Args:
        architecture (str): Architecture of the model.
        name (str): Name of the model to load.
        num_channels (int): Number of channels in the input.
        num_classes (int): Number of classes in the output layer.
        pretrained (str, optional): Pre-trained weights to load. Defaults to None.
        freeze_layers (bool): Whether to freeze the layers of the pre-trained model. Defaults to False.
    Returns:
        nn.Module: Model instance.
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model: {name}.")
    
    # CNN
    if architecture == "cnn":
        model: nn.Module = AVAILABLE_MODELS[name](weights=pretrained)

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

        if "resnet" in name:
            # Adjust the first convolutional layer to accept the specified number of input channels
            old_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels=num_channels,
                out_channels=old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False
            )
            # Adjust the output layer to match the number of classes
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
        
        elif "densenet" in name:
            # Adjust the first convolutional layer to accept the specified number of input channels
            old_features = model.features.conv0
            model.features.conv0 = nn.Conv2d(
                in_channels=num_channels,
                out_channels=old_features.out_channels,
                kernel_size=old_features.kernel_size,
                stride=old_features.stride,
                padding=old_features.padding,
                bias=False
            )
            # Adjust the output layer to match the number of classes
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, num_classes)

        elif "efficientnet" in name:
            # Adjust the first convolutional layer to accept the specified number of input channels
            old_features = model.features[0][0]
            model.features[0][0] = nn.Conv2d(
                in_channels=num_channels,
                out_channels=old_features.out_channels,
                kernel_size=old_features.kernel_size,
                stride=old_features.stride,
                padding=old_features.padding,
                bias=False
            )
            # Adjust the output layer to match the number of classes
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)

        else:
            raise ValueError(f"Model architecture not supported for: {name}.")

    else:
        raise ValueError(f"Invalid architecture: {architecture}. Use 'cnn' or 'transformer'.")

    return model