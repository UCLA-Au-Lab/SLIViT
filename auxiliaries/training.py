"""
Pure PyTorch training utilities for SLIViT.
No fastai dependency — provides training loop, checkpointing, early stopping,
logging, and data loading helpers.
"""

import os
import sys
import csv
import time
import random
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupShuffleSplit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Dataset resolution (mirrors auxiliaries/misc.py without fastai)
# ---------------------------------------------------------------------------

DATASET_NAME_TO_CLASS = {
    "xray2d": "MedMNISTDataset2D",
    "oct2d": "OCTDataset2D",
    "custom2d": "CustomDataset2D",
    "oct3d": "OCTDataset3D",
    "us3d": "USDataset3D",
    "mri3d": "MRIDataset3D",
    "ct3d": "MedMNISTDataset3D",
    "custom3d": "CustomDataset3D",
    "heidelberg": "HeidelbergOCTDataset",
}


def get_dataset_class(dataset_name):
    assert dataset_name in DATASET_NAME_TO_CLASS, (
        f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_NAME_TO_CLASS)}"
    )
    class_name = DATASET_NAME_TO_CLASS[dataset_name]
    module = __import__(f"datasets.{class_name}", fromlist=[class_name])
    return getattr(module, class_name)


# ---------------------------------------------------------------------------
# Data splitting (mirrors auxiliaries/misc.py without fastai)
# ---------------------------------------------------------------------------

def get_split_indices(meta, out_dir, split_ratio, label, split_col, pid_col):
    df = pd.read_csv(meta)

    if split_col in df.columns:
        logger.info(f"Using pre-defined split column: {split_col}")
        train_idx = np.argwhere(df[split_col].str.contains("train", case=False)).flatten()
        val_idx = np.argwhere(df[split_col].str.contains("val", case=False)).flatten()
        test_idx = np.argwhere(df[split_col].str.contains("test", case=False)).flatten()
        df.to_csv(f"{out_dir}/{os.path.split(meta)[-1]}", index=False)
        return train_idx, val_idx, test_idx

    train_ratio, val_ratio, test_ratio = split_ratio
    logger.info(f"Splitting dataset with ratio: {split_ratio}")

    script_name = os.path.basename(sys.argv[0]).split(".")[0]

    gss = GroupShuffleSplit(n_splits=1, train_size=train_ratio)
    train_idx, temp_idx = next(
        gss.split(df, y=None if script_name == "ssl_pretrain" else df[label], groups=df[pid_col])
    )

    if test_ratio == 0:
        val_idx = temp_idx
        test_idx = temp_idx[[]]
    else:
        test_val_df = df.iloc[temp_idx]
        gss_temp = GroupShuffleSplit(n_splits=1, train_size=val_ratio / sum(split_ratio[1:]))
        val_idx, test_idx = next(
            gss_temp.split(test_val_df, y=test_val_df[label], groups=test_val_df[pid_col])
        )
        val_idx = temp_idx[val_idx]
        test_idx = temp_idx[test_idx]

    df[split_col] = "train"
    df.loc[val_idx, split_col] = "val"
    df.loc[test_idx, split_col] = "test"
    df.to_csv(f"{out_dir}/{os.path.split(meta)[-1]}", index=False)

    return train_idx, val_idx, test_idx


def setup_ssl_dataloaders(args):
    """Build train/val PyTorch DataLoaders for self-supervised pre-training."""
    dataset_class = get_dataset_class(args.dataset_name)

    label = args.label if args.label is not None else []
    train_idx, val_idx, _ = get_split_indices(
        args.meta, args.out_dir, args.split_ratio, label, args.split_col, args.pid_col
    )

    dataset = dataset_class(
        args.meta,
        label or None,
        args.path_col,
        num_slices_to_use=args.slices,
        sparsing_method=args.sparsing_method,
        img_suffix=args.img_suffix,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.cpus,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.cpus,
        drop_last=False,
        pin_memory=True,
    )

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    return train_loader, val_loader


# Re-export MONAI's warmup cosine scheduler for convenience
from monai.optimizers.lr_scheduler import WarmupCosineSchedule


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Pure PyTorch training loop with mixed precision, early stopping, and checkpointing."""

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        device,
        out_dir,
        patience=5,
        min_delta=0.0,
        wandb_run=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.out_dir = out_dir
        self.patience = patience
        self.min_delta = min_delta
        self.wandb_run = wandb_run

        self.scaler = torch.amp.GradScaler()
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.best_epoch = 0

        self.csv_path = os.path.join(out_dir, "training_log.csv")
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr", "time_s"])

    def fit(self, train_loader, val_loader, epochs):
        self.model.to(self.device)
        logger.info(f"Training for {epochs} epochs on {self.device}")

        for epoch in range(epochs):
            t0 = time.time()

            train_loss = self._train_epoch(train_loader)
            val_loss = self._val_epoch(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            lr = self.optimizer.param_groups[0]["lr"]

            self._log(epoch, train_loss, val_loss, lr, elapsed)
            improved = self._checkpoint(epoch, val_loss)

            if self._early_stop():
                logger.info(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}"
                )
                break

        logger.info(f"Best model saved at epoch {self.best_epoch} with val loss {self.best_val_loss:.6f}")
        return self.best_val_loss

    def _train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda"):
                loss = self.model(x)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x, _ in loader:
                x = x.to(self.device, non_blocking=True)
                with torch.amp.autocast(device_type="cuda"):
                    loss = self.model(x)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def _checkpoint(self, epoch, val_loss):
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            torch.save(self.model.state_dict(), os.path.join(self.out_dir, "mae_best.pth"))
            return True
        else:
            self.epochs_without_improvement += 1
            return False

    def _early_stop(self):
        return self.epochs_without_improvement >= self.patience

    def _log(self, epoch, train_loss, val_loss, lr, elapsed):
        msg = (
            f"Epoch {epoch:03d} | "
            f"train_loss: {train_loss:.6f} | "
            f"val_loss: {val_loss:.6f} | "
            f"lr: {lr:.2e} | "
            f"time: {elapsed:.1f}s"
        )
        logger.info(msg)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{lr:.2e}", f"{elapsed:.1f}"])

        if self.wandb_run is not None:
            self.wandb_run.log(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr},
            )
