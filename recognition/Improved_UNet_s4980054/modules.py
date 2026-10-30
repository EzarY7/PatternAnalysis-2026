"""modules.py: U-Net models and loss for OASIS 2D brain MRI segmentation.
"""

import torch
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

class EncoderBlock(nn.Module):
    """Halve the height and width with max-pooling, then apply a DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)

class DecoderBlock(nn.Module):
    """Double the height and width, join the skip connection, then a DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    """Plain 2D U-Net (baseline).

    base_channels sets the width: the five levels use base, 2*base, 4*base,
    8*base and 16*base channels. The output has one channel per class.
    """

    def __init__(self, in_channels=1, num_classes=4, base_channels=16):
        super().__init__()
        c = [base_channels * 2 ** i for i in range(5)]
        self.inc = DoubleConv(in_channels, c[0])
        self.down1 = EncoderBlock(c[0], c[1])
        self.down2 = EncoderBlock(c[1], c[2])
        self.down3 = EncoderBlock(c[2], c[3])
        self.down4 = EncoderBlock(c[3], c[4])
        self.up1 = DecoderBlock(c[4], c[3], c[3])
        self.up2 = DecoderBlock(c[3], c[2], c[2])
        self.up3 = DecoderBlock(c[2], c[1], c[1])
        self.up4 = DecoderBlock(c[1], c[0], c[0])
        self.out = nn.Conv2d(c[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        return self.out(x)

class DiceLoss(nn.Module):
    """Multiclass soft Dice loss: 1 minus the mean Dice over the classes.

    logits: (B, C, H, W) raw model scores.
    target: (B, C, H, W) one-hot masks.
    """

    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3)  # sum over batch and pixels, keeping one value per class
        intersection = (probs * target).sum(dim=dims)
        total = probs.sum(dim=dims) + target.sum(dim=dims)
        dice = (2 * intersection + self.smooth) / (total + self.smooth)
        return 1 - dice.mean()

class ContextBlock(nn.Module):
    """Pre-activation residual block (Isensee et al.): two 3x3 convolutions with
    dropout in between, plus a shortcut that adds the input back."""

    def __init__(self, channels, dropout=0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.Dropout2d(dropout),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)

class ContextStage(nn.Module):
    """One level of the context pathway: a 3x3 convolution (stride 2 halves the
    image size, stride 1 keeps it) that sets the channel count, then a ContextBlock."""

    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.context = ContextBlock(out_channels, dropout)

    def forward(self, x):
        return self.context(self.conv(x))

class UpsampleBlock(nn.Module):
    """Decoder upsampling (Isensee et al.): repeat each pixel twice in each direction,
    then a 3x3 convolution that reduces the channel count (the paper halves it)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class LocalizationBlock(nn.Module):
    """Localization module (Isensee et al.): a 3x3 convolution that combines the
    concatenated features, then a 1x1 convolution that reduces the channel count."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class DecoderStage(nn.Module):
    """One level of the localization pathway: upsample, concatenate the matching
    encoder features, then combine them with a localization block."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = UpsampleBlock(in_channels, out_channels)
        self.localize = LocalizationBlock(skip_channels + out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.localize(x)