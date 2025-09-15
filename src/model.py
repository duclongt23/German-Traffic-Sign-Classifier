import torch
import torch.nn as nn
import torchvision.models as models

def build_model(n_classes = 43, backbone = 'resnet18', pretrained = True):
    if backbone == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        in_f = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes)
    )
        return model
    else:
        raise ValueError("Unsupported backbone")