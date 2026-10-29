"""modules.py: U-Net models and loss for OASIS 2D brain MRI segmentation.
"""

import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions, each followed by batch normalisation and ReLU.

    Keeps the image height and width; changes the channel count from
    in_channels to out_channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)