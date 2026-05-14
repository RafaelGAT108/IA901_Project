import torch.nn as nn
import torchvision.models as models

from transformers import ViTForImageClassification

AVAILABLE_MODELS = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
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
    "google/vit-base-patch16-224": ViTForImageClassification
}

# NOTE: Always verify the normalization and preprocessing steps required for each pre-trained model.

def load_model(architecture:str, name:str, num_classes:int, weights:str=None, freeze_layers:bool=False) -> nn.Module:
    """
    Load a model. The output layer is adjusted to match the number of classes.
    Args:
        architecture (str): Architecture of the model (CNN or Transformer).
        name (str): Name of the model to load.
        num_classes (int): Number of classes in the output layer.
        weights (str, Optional): Weights to load in the model. Defaults to None.
        freeze_layers (bool, Optional): Freeze the layers of the model. Defaults to False.
    Returns:
        nn.Module: Model instance.
    """
    if name not in AVAILABLE_MODELS:
        raise ValueError(f"Invalid model: {name}.")
    
    # CNN
    if architecture == "cnn":
        model = AVAILABLE_MODELS[name](weights=weights)

        if freeze_layers:
            for param in model.parameters():
                param.requires_grad = False

        # Adjust the output layer to match the number of classes
        if "resnet" in name:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

        elif "efficientnet" in name:
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    # Transformer
    elif architecture == "transformer":
        if ViTForImageClassification is None:
            raise ModuleNotFoundError("transformers is not installed, so transformer models cannot be loaded.")
        ModelClass = AVAILABLE_MODELS[name]
        model = ModelClass.from_pretrained(name,
                                           num_labels=num_classes, 
                                           ignore_mismatched_sizes=True)

    else:
        raise ValueError(f"Invalid architecture: {architecture}. Use 'cnn' or 'transformer'.")

    return model