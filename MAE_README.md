# MAE Self-Supervised Pre-training for SLIViT

Train SLIViT as an embedding model using Masked Autoencoders (MAE) on unlabeled 3D volumes. No labels required.

Uses [MONAI's MaskedAutoEncoderViT](https://docs.monai.io/en/stable/networks.html#maskedautoencodervit) under the hood.

## How it works

```
3D volume (N slices)
    |
    v
ConvNeXt feature extractor --> feature map (768, 8, N*8)
    |
    v
MONAI MAE: patch into N patches of (768, 8, 8) each
    |
    v
Randomly mask 75% of patches
    |
    v
ViT encoder (visible patches only) --> encoded representations
    |
    v
Lightweight decoder --> reconstruct masked patch features
    |
    v
MSE loss on masked positions
```

The encoder learns volumetric representations by predicting what masked slices look like from the visible ones. After training, the encoder's CLS token serves as a volume-level embedding.

Both the ConvNeXt feature extractor and the ViT encoder are trained end-to-end.

## Quick start

### 1. Create a config file

```yaml
# config.yaml
dataset_name: heidelberg
meta: ./meta/heidelberg.csv
out_dir: ./results/mae
fe_path: ./checkpoints/kermany/feature_extractor.pth
```

No `label` field needed -- it defaults to `null` for SSL.

### 2. Implement your dataset

Edit `datasets/HeidelbergOCTDataset.py` -- follow the shape contract in the docstring:

```
__getitem__ returns (volume, label) where:
  volume: FloatTensor of shape (3, 256, 256 * num_slices)
  label:  FloatTensor (ignored during SSL, use torch.tensor(0.0))
```

Each slice is resized to 256x256, converted to 3-channel float [0,1], then concatenated along width.

### 3. Train

```bash
python3 ssl_pretrain.py config.yaml
```

### 4. Extract embeddings

```python
from model.feature_extractor import get_feature_extractor
from model.mae import SLIViTMAE
import torch

mae = SLIViTMAE(
    feature_extractor=get_feature_extractor(4, "path/to/fe.pth"),
    num_patches=28,
)
mae.load_state_dict(torch.load("results/mae/mae_best.pth"))
mae.eval()

with torch.no_grad():
    embeddings = mae.encode(batch_of_volumes)  # (B, 256)
```

## Config reference

All fields are optional -- defaults are shown below.

```yaml
# --- Data ---
dataset_name: oct3d          # Dataset class (oct3d, heidelberg, us3d, etc.)
meta: null                   # Path to metadata CSV
label: null                  # Label column(s) -- null for SSL (no labels needed)
path_col: path               # CSV column with volume paths
split_col: split             # CSV column for train/val/test split
pid_col: pid                 # CSV column for patient ID (grouped splitting)
split_ratio: [0.85, 0.15, 0] # Train/val/test ratio
slices: 28                   # Number of slices per volume
sparsing_method: eq          # Slice selection: eq (equally-spaced), mid (middle-focused)
img_suffix: tiff             # Slice file extension

# --- Feature extractor ---
fe_path: ""                  # Path to pretrained ConvNeXt weights (empty = ImageNet)
fe_classes: 4                # Number of classes the feature extractor was pretrained on

# --- Encoder (MONAI ViT) ---
hidden_size: 256             # Transformer embedding dimension
num_layers: 5                # Number of transformer layers
num_heads: 16                # Number of attention heads (must divide hidden_size)
mlp_dim: 1024                # Feedforward dimension
dropout: 0.0
pos_embed_type: sincos       # Positional embedding: sincos or learnable

# --- Decoder ---
decoder_hidden_size: 128     # Decoder embedding dimension
decoder_num_layers: 2        # Decoder transformer layers
decoder_num_heads: 4         # Decoder attention heads (must divide decoder_hidden_size)
decoder_mlp_dim: 512         # Decoder feedforward dimension

# --- MAE ---
mask_ratio: 0.75             # Fraction of patches to mask (0.75 = 75%)

# --- Training ---
out_dir: ./results/mae
epochs: 100
batch_size: 4
lr: 1.5e-4
weight_decay: 0.05
warmup_epochs: 10            # Linear warmup before cosine decay
patience: 5                  # Early stopping patience
min_delta: 0.0               # Minimum improvement for early stopping
seed: 1
gpu_id: "0"
cpus: 16
wandb_name: null             # Set to enable Weights & Biases logging
```

## Outputs

```
{out_dir}/
  mae_best.pth          # Best model (lowest val loss)
  mae_final.pth         # Final model (last epoch)
  config.yaml           # Full config snapshot for reproducibility
  training_log.csv      # Per-epoch train/val loss, lr, time
  ssl_pretrain.log      # Full training log
```

## Architecture details

| Component | Config | Default |
|-----------|--------|---------|
| Feature extractor | ConvNeXt-Tiny | `facebook/convnext-tiny-224` |
| Encoder | MONAI ViT | hidden_size=256, layers=5, heads=16 |
| Decoder | MONAI ViT (lightweight) | hidden_size=128, layers=2, heads=4 |
| Patch feature dim | 768 x 8 x 8 = 49,152 | Fixed by ConvNeXt output |
| Embedding dim | = hidden_size | 256 |
| Positional embedding | sincos | Fixed (not learned) |

The ConvNeXt output (768, 8, N*8) is treated as a 2D image with 768 channels. MONAI's MAE patches it with (8, 8) patches, yielding N patches -- one per B-scan slice.

## Downstream use

After pre-training, the encoder can be used in two ways:

**As a frozen embedding model** -- extract CLS token embeddings and train a linear probe or downstream model on top:

```python
embeddings = mae.encode(volumes)  # (B, 256)
```

**For fine-tuning** -- load the pre-trained encoder weights into a supervised model and fine-tune with labels on a downstream task.

## Training tips

- **Mask ratio 0.75** works well because adjacent OCT slices are highly correlated. If your data has less inter-slice redundancy, try 0.5-0.6.
- **Batch size 4** is typical for 3D volumes. The MAE loss doesn't depend on batch size (unlike contrastive methods), so small batches are fine.
- **num_heads must divide hidden_size**. Default: 256 / 16 = 16 per head. If you change hidden_size, pick num_heads accordingly.
- **sincos positional embeddings** (default) are fixed and work well for small datasets. Switch to `learnable` if you have lots of data.
- **Monitor val loss** -- it should decrease steadily. If it plateaus at epoch 0, the learning rate is likely too high.
- **Mixed precision** is enabled by default (`torch.amp`), roughly halving memory usage.
