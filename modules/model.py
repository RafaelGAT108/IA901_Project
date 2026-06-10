"""
Model loading and architecture definition for lung sound classification.
"""

import torch.nn as nn
import torchvision.models as models

AVAILABLE_CNNS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "densenet121": models.densenet121,
    "densenet161": models.densenet161,
    "densenet169": models.densenet169,
    "densenet201": models.densenet201,
    "efficientnet_b0": models.efficientnet_b0,
    "efficientnet_b1": models.efficientnet_b1,
    "efficientnet_b2": models.efficientnet_b2,
    "efficientnet_b3": models.efficientnet_b3,
    "efficientnet_b4": models.efficientnet_b4,
    "efficientnet_b5": models.efficientnet_b5,
    "efficientnet_b6": models.efficientnet_b6,
    "efficientnet_b7": models.efficientnet_b7,
    "efficientnet_v2_s": models.efficientnet_v2_s,
    "efficientnet_v2_m": models.efficientnet_v2_m,
    "efficientnet_v2_l": models.efficientnet_v2_l,
    "mobilenet_v2": models.mobilenet_v2,
    "mobilenet_v3_small": models.mobilenet_v3_small,
    "mobilenet_v3_large": models.mobilenet_v3_large,
    "vgg11": models.vgg11,
    "vgg13": models.vgg13,
    "vgg16": models.vgg16,
    "vgg19": models.vgg19,
    "vgg11_bn": models.vgg11_bn,
    "vgg13_bn": models.vgg13_bn,
    "vgg16_bn": models.vgg16_bn,
    "vgg19_bn": models.vgg19_bn,
    "inception_v3": models.inception_v3,
}

# NOTE: Always verify the normalization and preprocessing steps required for each pre-trained model.

class LungSoundModel:
    """ Class for loading and customizing pre-trained models for lung sound classification. """
    def __init__(
            self,
            name: str,
            num_channels: int,
            num_classes: int,
            pretrained: str = None,
            freeze_layers: bool = False
        ):
        """
        Load a model from torchvision with the specified parameters.
        Args:
            name (str): Name of the model to load.
            num_channels (int): Number of channels in the input.
            num_classes (int): Number of classes in the output layer.
            pretrained (str, optional): Pre-trained weights to load. Defaults to None.
            freeze_layers (bool): Whether to freeze the layers of the pre-trained model. Defaults to False.
        """
        if name not in AVAILABLE_CNNS:
            raise ValueError(f"Invalid model: {name}.")

        self.name = name
        self.in_channels = num_channels
        self.out_channels = num_classes
        self.weights = pretrained
        self.freeze_layers = freeze_layers
        self.model = self.load_model()
        
    def load_model(self) -> nn.Module:
        # CNN
        if self.name in AVAILABLE_CNNS:
            model: nn.Module = AVAILABLE_CNNS[self.name](weights=self.weights)

            if "resnet" in self.name:
                # Adjust the first convolutional layer to accept the specified number of input channels
                old_conv = model.conv1
                model.conv1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
                self.input_layer = model.conv1
                # Adjust the output layer to match the number of classes
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.out_channels)
                self.output_layer = model.fc
            
            elif "densenet" in self.name:
                # Adjust the first convolutional layer to accept the specified number of input channels
                old_features = model.features.conv0
                model.features.conv0 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=old_features.out_channels,
                    kernel_size=old_features.kernel_size,
                    stride=old_features.stride,
                    padding=old_features.padding,
                    bias=False
                )
                self.input_layer = model.features.conv0
                # Adjust the output layer to match the number of classes
                num_ftrs = model.classifier.in_features
                model.classifier = nn.Linear(num_ftrs, self.out_channels)
                self.output_layer = model.classifier

            elif "efficientnet" in self.name:
                # Adjust the first convolutional layer to accept the specified number of input channels
                old_features = model.features[0][0]
                model.features[0][0] = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=old_features.out_channels,
                    kernel_size=old_features.kernel_size,
                    stride=old_features.stride,
                    padding=old_features.padding,
                    bias=False
                )
                self.input_layer = model.features[0][0]
                # Adjust the output layer to match the number of classes
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, self.out_channels)
                self.output_layer = model.classifier[1]

            elif "mobilenet_v2" in self.name:
                # Adjust the first convolutional layer to accept the specified number of input channels
                old_features = model.features[0][0]
                model.features[0][0] = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=old_features.out_channels,
                    kernel_size=old_features.kernel_size,
                    stride=old_features.stride,
                    padding=old_features.padding,
                    bias=False
                )
                self.input_layer = model.features[0][0]
                # Adjust the output layer to match the number of classes
                num_ftrs = model.classifier[1].in_features
                model.classifier[1] = nn.Linear(num_ftrs, self.out_channels)
                self.output_layer = model.classifier[1]

            elif "mobilenet_v3" in self.name:
                # Adjust the first convolutional layer to accept the specified number of input channels
                old_features = model.features[0][0]
                model.features[0][0] = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=old_features.out_channels,
                    kernel_size=old_features.kernel_size,
                    stride=old_features.stride,
                    padding=old_features.padding,
                    bias=False
                )
                self.input_layer = model.features[0][0]
                # Adjust the output layer to match the number of classes
                num_ftrs = model.classifier[3].in_features
                model.classifier[3] = nn.Linear(num_ftrs, self.out_channels)
                self.output_layer = model.classifier[3]

            elif "vgg" in self.name:
                # Adjust the first convolutional layer to accept the specified number of input channels
                old_features = model.features[0]
                model.features[0] = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=old_features.out_channels,
                    kernel_size=old_features.kernel_size,
                    stride=old_features.stride,
                    padding=old_features.padding,
                    bias=False
                )
                self.input_layer = model.features[0]
                # Adjust the output layer to match the number of classes
                num_ftrs = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_ftrs, self.out_channels)
                self.output_layer = model.classifier[6]

            elif self.name == "inception_v3":
                # Adjust the first convolutional layer to accept the specified number of input channels
                old_conv = model.Conv2d_1a_3x3.conv
                model.Conv2d_1a_3x3.conv = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=old_conv.padding,
                    bias=False
                )
                self.input_layer = model.Conv2d_1a_3x3.conv
                # Adjust the main output layer to match the number of classes
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.out_channels)
                # Disable auxiliary logits for simplicity (can be enabled if needed)
                model.aux_logits = False
                model.AuxLogits = None
                self.output_layer = model.fc

            else:
                raise ValueError(f"Model architecture not supported for: {self.name}.")

        else:
            raise ValueError(f"Invalid architecture: {self.name}.")

        # NOTE: Here, we are freezing all layers except the input and output layers.
        # This is an experimental choice since both the in_channels and out_channels can be different from the pre-trained model.
        if self.freeze_layers:
            # Freeze all layers
            for param in model.parameters():
                param.requires_grad = False
            # Unfreeze the input layer
            for param in self.input_layer.parameters():
                param.requires_grad = True
            # Unfreeze the output layer
            for param in self.output_layer.parameters():
                param.requires_grad = True

        # The parameters to be passed to the optimizer
        # Only include parameters that require gradients (i.e. unfrozen layers)
        self.trainable_params = [p for p in model.parameters() if p.requires_grad]

        return model