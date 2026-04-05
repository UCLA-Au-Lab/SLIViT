"""
Skeleton dataset for Heidelberg OCT 3D volumes.

Fill in the TODO sections with your data loading logic. The shape contract
documented below must be satisfied for the MAE training pipeline to work.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tf
import pandas as pd
import zarr

# Per-slice transform: grayscale → (3, 256, 256) float tensor in [0, 1]
per_slice_transform = tf.Compose([
    tf.ToPILImage(),
    tf.Resize((256, 256)),
    tf.ToTensor(),                              # → (1, 256, 256), float [0, 1]
    tf.Lambda(lambda x: x.expand(3, -1, -1)),   # → (3, 256, 256)
])


class HeidelbergOCTDataset(Dataset):
    """
    Dataset for Heidelberg OCT 3D volumes.

    Each volume is a stack of 2D grayscale B-scan slices. This dataset
    loads N equally-spaced slices, transforms each to (3, 256, 256),
    and concatenates them along the width dimension.

    Shape contract
    ──────────────
    __getitem__ returns (volume, label) where:

        volume: torch.FloatTensor, shape (3, 256, 256 * num_slices)
                ┌─────────────────────────────────────────────────────┐
                │ slice_0 │ slice_1 │ ... │ slice_{N-1}              │
                │(3,256,256)(3,256,256)    (3,256,256)               │
                │         concatenated along dim=-1 (width)          │
                └─────────────────────────────────────────────────────┘
                This enters ConvNeXt → (768, H, W) where H*W = N*64.
                The model reshapes to (N, 768, 64) — one patch per slice.

        label:  torch.FloatTensor, shape (1,) or scalar.
                Ignored during SSL pre-training (the Trainer does
                `for x, _ in loader`), but the DataLoader still
                yields it. Return a dummy torch.tensor(0.0) for SSL.

    Constraints
    ───────────
    - num_slices must match the `slices` config value (default 28).
      If a volume has more slices than needed, subsample (e.g. equally-spaced).
      If fewer, pad by repeating edge slices.
    - Pixel values must be float32 in [0, 1]. (ToTensor() handles this
      for uint8 inputs; if your slices are already float, normalize manually.)
    - 3-channel expansion is required — ConvNeXt expects RGB input.
    """

    def __init__(
        self,
        meta,          # str (path to CSV) or pd.DataFrame
        label_name,    # list[str] — column name(s) for labels
        path_col_name, # str — column name containing volume paths
        num_slices_to_use=28,
        sparsing_method="eq",
        img_suffix="tiff",
    ):
        #   - Read the CSV / dataframe
        #   - Store volume paths and labels
        #   - Example:
        
        df = pd.read_csv(meta) if isinstance(meta, str) else meta
        self.paths = df[path_col_name].values
        
        
        self.num_slices = num_slices_to_use
        self.sparsing_method = sparsing_method
        self.img_suffix = img_suffix
        self.transform = per_slice_transform

        # raise NotImplementedError("Fill in __init__ with your metadata loading logic")

    def __len__(self):
        return len(self.paths)
        # raise NotImplementedError

    def __getitem__(self, idx):

        # ── Step 1: Get the volume path ──
        vol_path = self.paths[idx]

        # ── Step 2: List and select slices ──
        # Get all slice filenames, sort them, and pick N equally-spaced indices.
        #
        # import os
        # all_slices = sorted([
        #     f for f in os.listdir(vol_path) if f.endswith(self.img_suffix)
        # ])
        grp = zarr.open_group("/hdd1/UCLA_MP/volumetric/arrs.zarr", path=vol_path)
        vol = grp["volume"] # shape of (C, H, W, )
        # total = len(all_slices)
        total = vol.shape[1]
        indices = np.linspace(0, total - 1, self.num_slices).astype(int)


        # ── Step 3: Load and transform each slice ──
        # Each slice should be loaded as a numpy array or PIL Image.
        # The transform handles: resize → float tensor → 3-channel.
        #
        # 
        # from torchvision.transforms import ToTensor
        slices = []
        for i in indices:
            img = vol[:, i, :, :]
            slices.append(self.transform(img))
            # Each slice is now (3, 256, 256)

        # ── Step 4: Concatenate along width ──
        volume = torch.cat(slices, dim=-1)  # → (3, 256, 256 * num_slices)

        # ── Step 5: Get label (dummy for SSL) ──
        label = torch.tensor(0.0)  # unused during SSL
        # # For supervised: label = torch.FloatTensor(self.labels[idx])

        return volume, label

        # raise NotImplementedError("Fill in __getitem__ with your slice loading logic")
