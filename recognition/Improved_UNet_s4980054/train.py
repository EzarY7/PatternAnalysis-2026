"""train.py: train a 2D U-Net on the OASIS brain MRI slices.

Usage:
    python train.py --root ~/oasis_data --model unet --base-channels 16 --epochs 30
"""

import argparse
import random

import torch
import torch.nn.functional as F

from dataset import NUM_CLASSES
from modules import UNet

MODELS = {"unet": UNet}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model on OASIS.")
    parser.add_argument("--root", required=True,
                        help="folder holding the six keras_png_slices_* folders")
    parser.add_argument("--model", choices=MODELS, default="unet")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    parser.add_argument("--max-batches", type=int, default=None,
                        help="limit batches per epoch (for quick debugging)")
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(name, base_channels):
    return MODELS[name](base_channels=base_channels)

def dice_counts(logits, target):
    """Per-class overlap counts for one batch, from hard predictions.

    logits: (B, C, H, W) raw model scores.
    target: (B, C, H, W) one-hot masks.
    Returns (intersection, total), each a tensor with one number per class.
    """
    num_classes = logits.shape[1]
    pred = F.one_hot(logits.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)  # add up over the batch and all pixels, keep one value per class
    intersection = (pred * target).sum(dim=dims)
    total = pred.sum(dim=dims) + target.sum(dim=dims)
    return intersection, total


def dice_from_counts(intersection, total, smooth=1e-5):
    """Per-class Dice score from counts added up over a whole dataset."""
    return (2 * intersection + smooth) / (total + smooth)

@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    """Per-class Dice over a whole dataset (no weight updates)."""
    model.eval()
    intersection = torch.zeros(NUM_CLASSES, device=device)
    total = torch.zeros(NUM_CLASSES, device=device)
    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)
        inter, tot = dice_counts(model(images), masks)
        intersection += inter
        total += tot
    return dice_from_counts(intersection, total).cpu()

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    model = build_model(args.model, args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | model: {args.model} | "
          f"base channels: {args.base_channels} | parameters: {n_params:,}")


if __name__ == "__main__":
    main()