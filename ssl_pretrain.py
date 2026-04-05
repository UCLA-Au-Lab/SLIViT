"""
Self-supervised MAE pre-training for SLIViT on 3D volumes.

Trains a Masked Autoencoder end-to-end (ConvNeXt + ViT encoder + decoder)
to learn volumetric representations without labels.

Usage:
    python3 ssl_pretrain.py config.yaml
"""

import os
import sys
import logging
import warnings
from types import SimpleNamespace

import yaml
import torch

from model.feature_extractor import get_feature_extractor
from model.mae import SLIViTMAE
from auxiliaries.training import (
    set_seed,
    setup_ssl_dataloaders,
    Trainer,
    WarmupCosineScheduler,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Data
    "dataset_name": "oct3d",
    "meta": None,
    "label": ["dummy"],
    "path_col": "path",
    "split_col": "split",
    "pid_col": "pid",
    "split_ratio": [0.85, 0.15, 0],
    "slices": 28,
    "sparsing_method": "eq",
    "img_suffix": "tiff",
    # Feature extractor
    "fe_path": "",
    "fe_classes": 4,
    # Encoder (matches SLIViT ViT defaults)
    "vit_dim": 256,
    "vit_depth": 5,
    "heads": 20,
    "dropout": 0.0,
    "emb_dropout": 0.0,
    # Decoder
    "decoder_dim": 128,
    "decoder_depth": 2,
    "decoder_heads": 4,
    # MAE
    "mask_ratio": 0.75,
    # Training
    "out_dir": "./results/mae",
    "epochs": 100,
    "batch_size": 4,
    "lr": 1.5e-4,
    "weight_decay": 0.05,
    "warmup_epochs": 10,
    "patience": 5,
    "min_delta": 0.0,
    "seed": 1,
    "gpu_id": "0",
    "cpus": 16,
    "wandb_name": None,
}


def load_config(path):
    with open(path) as f:
        user_cfg = yaml.safe_load(f) or {}

    # Ensure label is a list
    if "label" in user_cfg and isinstance(user_cfg["label"], str):
        user_cfg["label"] = user_cfg["label"].split(",")

    # Ensure split_ratio is a list of floats
    if "split_ratio" in user_cfg and isinstance(user_cfg["split_ratio"], str):
        user_cfg["split_ratio"] = [float(x) for x in user_cfg["split_ratio"].split(",")]

    cfg = {**DEFAULTS, **user_cfg}
    return SimpleNamespace(**cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(f"Usage: python3 {sys.argv[0]} config.yaml")
        sys.exit(0 if "--help" in sys.argv else 1)

    args = load_config(sys.argv[1])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(args.out_dir, "ssl_pretrain.log"), mode="w"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)

    # Save config to output dir for reproducibility
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f, default_flow_style=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {args.out_dir}")
    logger.info(f"Config: {vars(args)}")

    # Data
    train_loader, val_loader = setup_ssl_dataloaders(args)

    # Model
    feature_extractor = get_feature_extractor(args.fe_classes, args.fe_path)

    mae = SLIViTMAE(
        feature_extractor=feature_extractor,
        num_patches=args.slices,
        encoder_dim=args.vit_dim,
        encoder_depth=args.vit_depth,
        encoder_heads=args.heads,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        decoder_heads=args.decoder_heads,
        mask_ratio=args.mask_ratio,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
    )

    n_params = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    logger.info(f"MAE parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        mae.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler: linear warmup + cosine decay
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=1e-6,
    )

    # Wandb (optional)
    wandb_run = None
    if args.wandb_name is not None:
        import wandb
        wandb_run = wandb.init(project=args.wandb_name, config=vars(args))

    # Train
    trainer = Trainer(
        model=mae,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        out_dir=args.out_dir,
        patience=args.patience,
        min_delta=args.min_delta,
        wandb_run=wandb_run,
    )

    try:
        trainer.fit(train_loader, val_loader, args.epochs)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        torch.save(mae.state_dict(), os.path.join(args.out_dir, "mae_final.pth"))
        logger.info(f"Final checkpoint saved to {args.out_dir}/mae_final.pth")

        if wandb_run is not None:
            wandb_run.finish()

    logger.info("Done.")
