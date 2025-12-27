"""
Model definitions for loading saved PyTorch models.
These classes must match the definitions used when the models were saved.
"""
import torch.nn as nn


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for ResNet.
    This class must be defined before loading models that use it.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

