"""predict.py: score trained segmentation models on the OASIS test set.

Usage:
    python predict.py --root ~/oasis_data --model unet --weights runs/unet_b16/best.pt
"""

import argparse

import torch

from dataset import get_dataloaders
from train import MODELS, build_model, get_device, evaluate

CLASS_NAMES = ["background", "CSF", "grey matter", "white matter"]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the OASIS test set.")
    parser.add_argument("--root", required=True,
                        help="folder holding the six keras_png_slices_* folders")
    parser.add_argument("--model", choices=MODELS, default="unet")
    parser.add_argument("--weights", required=True, help="path to best.pt from train.py")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def load_model(name, base_channels, weights, device):
    """Build a model, load trained weights into it and switch to evaluation mode."""
    model = build_model(name, base_channels).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main():
    args = parse_args()
    device = get_device()
    model = load_model(args.model, args.base_channels, args.weights, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | model: {args.model} | parameters: {n_params:,}")

    _, _, test_loader = get_dataloaders(args.root, batch_size=args.batch_size,
                                        num_workers=args.num_workers)
    print(f"test set: {len(test_loader.dataset)} slices, {len(test_loader)} batches")

    dice = evaluate(model, test_loader, device)
    print("test Dice per class:")
    for name, d in zip(CLASS_NAMES, dice.tolist()):
        print(f"  {name:12s} {d:.4f}")
    print(f"  mean         {dice.mean().item():.4f}")
    print(f"  lowest label {dice.min().item():.4f} | all labels >= 0.9: {bool((dice >= 0.9).all())}")


if __name__ == "__main__":
    main()