"""predict.py: score trained segmentation models on the OASIS test set.

Usage:
    python predict.py --root ~/oasis_data --unet-weights runs/unet_b16/best.pt --improved-weights runs/improved_unet_b16/best.pt
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
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--unet-weights", help="best.pt of the plain U-Net")
    parser.add_argument("--improved-weights", help="best.pt of the Improved U-Net")
    args = parser.parse_args()
    if args.unet_weights is None and args.improved_weights is None:
        parser.error("give at least one of --unet-weights and --improved-weights")
    return args


def load_model(name, base_channels, weights, device):
    """Build a model, load trained weights into it and switch to evaluation mode."""
    model = build_model(name, base_channels).to(device)
    state = torch.load(weights, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def print_comparison(results):
    """Print per-class test Dice for each model side by side."""
    names = list(results)
    two = len(names) == 2

    def row(label, values):
        line = f"{label:14s}" + "".join(f"{v:16.4f}" for v in values)
        if two:
            line += f"{values[1] - values[0]:+16.4f}"
        print(line)

    print(f"{'':14s}" + "".join(f"{n:>16s}" for n in names)
          + (f"{'improved-unet':>16s}" if two else ""))
    for i, class_name in enumerate(CLASS_NAMES):
        row(class_name, [results[n][i].item() for n in names])
    row("mean", [results[n].mean().item() for n in names])
    row("lowest label", [results[n].min().item() for n in names])
    for n in names:
        print(f"{n}: all labels >= 0.9: {bool((results[n] >= 0.9).all())}")

def main():
    args = parse_args()
    device = get_device()
    _, _, test_loader = get_dataloaders(args.root, batch_size=args.batch_size,
                                        num_workers=args.num_workers)
    print(f"device: {device} | test set: {len(test_loader.dataset)} slices")

    results = {}
    for name, weights in (("unet", args.unet_weights),
                          ("improved_unet", args.improved_weights)):
        if weights is None:
            continue
        model = load_model(name, args.base_channels, weights, device)
        results[name] = evaluate(model, test_loader, device)
    print_comparison(results)


if __name__ == "__main__":
    main()