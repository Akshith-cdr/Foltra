"""ImageNet-initialized ResNet-18 with a dataset-specific classifier."""
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes, pretrained=True):
    if num_classes < 2:
        raise ValueError("Classification requires at least two classes")
    model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
