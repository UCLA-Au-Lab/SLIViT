"""
2D slice dataset for Heidelberg OCT volumes.

Loads individual B-scan slices directly from zarr — one slice per __getitem__
call, no full volume loading. Used for Stage 1 (2D MAE pretraining).
"""

import random

import torch
import numpy as np
import pandas as pd
import zarr
from torch.utils.data import Dataset
from torchvision import transforms as tf


per_slice_transform = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    tf.ToTensor(),                              # → (1, 256, 256), float [0, 1]
    tf.Lambda(lambda x: x.expand(3, -1, -1)),   # → (3, 256, 256)
])

ZARR_STORE = "/hdd1/UCLA_MP/volumetric/arrs.zarr"


class HeidelbergOCTSliceDataset(Dataset):
    """
    Yields individual 2D B-scan slices (3, 256, 256) from Heidelberg OCT zarr volumes.

    Each volume has shape (1, num_slices, H, W) in zarr. This dataset reads
    one random slice per item — no full volume loading.

    Total dataset length = num_volumes × slices_per_volume.
    Each epoch samples different random slices.
    """

    def __init__(
        self,
        meta,
        label_name,
        path_col_name,
        num_slices_to_use=28,
        slices_per_volume=10,
        sparsing_method="eq",
        img_suffix="tiff",
    ):
        df = pd.read_csv(meta) if isinstance(meta, str) else meta
        self.paths = df[path_col_name].values
        self.slices_per_volume = slices_per_volume
        self.transform = per_slice_transform

        # Cache slice counts per volume to avoid repeated zarr metadata reads
        self._slice_counts = None

    def _get_slice_count(self, vol_path):
        grp = zarr.open_group(ZARR_STORE, path=vol_path)
        return grp["volume"].shape[1]

    def __len__(self):
        return len(self.paths) * self.slices_per_volume

    def __getitem__(self, idx):
        vol_idx = idx // self.slices_per_volume

        vol_path = self.paths[vol_idx]
        grp = zarr.open_group(ZARR_STORE, path=vol_path)
        vol = grp["volume"]

        # Pick a random slice
        total_slices = vol.shape[1]
        slice_idx = random.randint(0, total_slices - 1)

        # Load single slice directly — (H, W)
        img = vol[0, slice_idx, :, :]

        # Transform: resize → tensor → 3-channel → (3, 256, 256)
        img = self.transform(img)

        return img, torch.tensor(0.0)
