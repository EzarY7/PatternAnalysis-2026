"""dataset.py: OASIS 2D brain MRI data loading for segmentation.

The OASIS data arrives pre-split into train / validate / test folders, with one
PNG per 2D slice and a matching mask PNG:

    keras_png_slices_train/case_001_slice_0.nii.png
    keras_png_slices_seg_train/seg_001_slice_0.nii.png

Masks hold the grey values 0, 85, 170, 255. Dividing by 85 gives the class
indices 0, 1, 2, 3.
"""

import re
from pathlib import Path
from itertools import combinations

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

NUM_CLASSES = 4
MASK_STEP = 85  # mask grey levels 0, 85, 170, 255 -> classes 0, 1, 2, 3

# split name -> (image folder, mask folder)
SPLIT_FOLDERS = {
    "train": ("keras_png_slices_train", "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test": ("keras_png_slices_test", "keras_png_slices_seg_test"),
}

_NAME_RE = re.compile(r"case_(\d+)_slice_(\d+)")

def list_pairs(root, split):
    """Return a sorted list of (image_path, mask_path) for one split."""
    img_folder, seg_folder = SPLIT_FOLDERS[split]
    img_dir = Path(root).expanduser() / img_folder
    seg_dir = Path(root).expanduser() / seg_folder

    pairs = []
    for img_path in sorted(img_dir.glob("case_*.png")):
        if _NAME_RE.match(img_path.name) is None:
            continue
        # case_001_slice_0.nii.png -> seg_001_slice_0.nii.png
        seg_path = seg_dir / img_path.name.replace("case_", "seg_", 1)
        if not seg_path.exists():
            raise FileNotFoundError(f"No mask for {img_path.name}: expected {seg_path}")
        pairs.append((img_path, seg_path))

    if not pairs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return pairs

def case_ids(root, split):
    """Set of case (patient) IDs present in a split."""
    return {_NAME_RE.match(img.name).group(1) for img, _ in list_pairs(root, split)}


def assert_no_case_overlap(root):
    """Raise an error if any case appears in more than one split.

    The data is already split, so this is the leakage check: all slices of a
    patient must stay in one split. Returns the number of cases per split.
    """
    ids = {split: case_ids(root, split) for split in SPLIT_FOLDERS}
    for a, b in combinations(ids, 2):
        shared = ids[a] & ids[b]
        if shared:
            raise ValueError(
                f"Case leakage between {a} and {b}: "
                f"{len(shared)} shared cases, e.g. {sorted(shared)[:5]}"
            )
    return {split: len(case_set) for split, case_set in ids.items()}

def load_image(path):
    """Greyscale image as a (H, W) uint8 array."""
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)

def load_mask(path):
    """Mask as a (H, W) uint8 array of class indices 0..NUM_CLASSES-1."""
    raw = np.array(Image.open(path).convert("L"), dtype=np.int16)
    classes = np.rint(raw / MASK_STEP).astype(np.int16)
    if classes.max() >= NUM_CLASSES or not np.array_equal(classes * MASK_STEP, raw):
        raise ValueError(
            f"Unexpected mask values in {path}: {np.unique(raw).tolist()}"
        )
    return classes.astype(np.uint8)

def to_one_hot(mask, num_classes=NUM_CLASSES):
    """(H, W) long class indices -> (C, H, W) float one-hot."""
    return torch.nn.functional.one_hot(mask, num_classes).permute(2, 0, 1).float()

class OASISDataset(Dataset):
    """One split of the OASIS 2D slices.

    Each item is (image, mask):
        image: float32 (1, H, W), scaled to [0, 1]
        mask:  float32 (NUM_CLASSES, H, W) one-hot if one_hot=True,
               otherwise int64 (H, W) class indices
    """

    def __init__(self, root, split, one_hot=True):
        self.pairs = list_pairs(root, split)
        self.one_hot = one_hot

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, seg_path = self.pairs[idx]
        image = torch.from_numpy(load_image(img_path)).unsqueeze(0)  # (1, H, W)
        image = image.float() / 255.0
        mask = torch.from_numpy(load_mask(seg_path)).long()  # (H, W)
        if self.one_hot:
            mask = to_one_hot(mask)
        return image, mask