"""dataset.py: OASIS 2D brain MRI data loading for segmentation.

The OASIS data arrives pre-split into train / validate / test folders, with one
PNG per 2D slice and a matching mask PNG:

    keras_png_slices_train/case_001_slice_0.nii.png
    keras_png_slices_seg_train/seg_001_slice_0.nii.png

Masks hold the grey values 0, 85, 170, 255. Dividing by 85 gives the class
indices 0, 1, 2, 3.
"""

import re

NUM_CLASSES = 4
MASK_STEP = 85  # mask grey levels 0, 85, 170, 255 -> classes 0, 1, 2, 3

# split name -> (image folder, mask folder)
SPLIT_FOLDERS = {
    "train": ("keras_png_slices_train", "keras_png_slices_seg_train"),
    "validate": ("keras_png_slices_validate", "keras_png_slices_seg_validate"),
    "test": ("keras_png_slices_test", "keras_png_slices_seg_test"),
}

_NAME_RE = re.compile(r"case_(\d+)_slice_(\d+)")