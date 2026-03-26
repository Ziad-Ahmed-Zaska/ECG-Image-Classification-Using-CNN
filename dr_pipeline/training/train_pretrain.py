"""
Stage 2 training: encoder pretraining on EyePACS.

Supports three variants (Section 4.3):
  A - Self-supervised (SimCLR) only
  B - Supervised DR grading only
  C - Self-supervised then supervised fine-tune

Usage
-----
    python -m dr_pipeline.training.train_pretrain --variant A
    python -m dr_pipeline.training.train_pretrain --variant B
    python -m dr_pipeline.training.train_pretrain --variant C
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dr_pipeline.config import (
    CHECKPOINT_DIR,
    EYEPACS_CONFIG,
    LOG_DIR,
    TrainConfig,
    ensure_dirs,
)
from dr_pipeline.datasets.dataset_manifest import DatasetManifest
from dr_pipeline.datasets.eyepacs import EyePACSDataset
from dr_pipeline.models.dr_grading import DRGradingModel
from dr_pipeline.models.encoder import DREncoder
from dr_pipeline.models.self_supervised import SimCLRPretrainer

logger = logging.getLogger(__name__)


def _setup_logging(variant: str) -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / f"pretrain_variant_{variant}.log"),
        ],
    )


# ------------------------------------------------------------------
# Variant A: self-supervised pretraining
# ------------------------------------------------------------------

def train_ssl(
    cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
) -> DREncoder:
    """
    Train an encoder with SimCLR on unlabelled EyePACS images.

    Returns the encoder with learned weights (projection head discarded).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = DREncoder(
        backbone_name=cfg.encoder_name,
        pretrained_imagenet=cfg.pretrained_imagenet,
    )
    model = SimCLRPretrainer(
        encoder=encoder,
        projection_dim=cfg.ssl_projection_dim,
        temperature=cfg.ssl_temperature,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.ssl_epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

    best_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.ssl_epochs + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (view1, view2) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                loss, _, _ = model(view1, view2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - t0
        logger.info(
            "SSL Epoch %d/%d  loss=%.4f  lr=%.2e  time=%.1fs",
            epoch, cfg.ssl_epochs, avg_loss,
            scheduler.get_last_lr()[0], elapsed,
        )

        # Simple early stopping on training loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            model.save_encoder(str(CHECKPOINT_DIR / "encoder_ssl_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return model.get_encoder()


# ------------------------------------------------------------------
# Variant B: supervised DR grading
# ------------------------------------------------------------------

def train_supervised(
    cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder: Optional[DREncoder] = None,
) -> DREncoder:
    """
    Fine-tune encoder on EyePACS DR grading labels.

    If *encoder* is provided (Variant C), it was pretrained with SSL.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DRGradingModel(encoder=encoder).to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        train_loss /= max(total, 1)
        train_acc = correct / max(total, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_loss)
        logger.info(
            "DR-Grade Epoch %d/%d  train_loss=%.4f  train_acc=%.3f  "
            "val_loss=%.4f  val_acc=%.3f",
            epoch, cfg.epochs, train_loss, train_acc, val_loss, val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save_encoder(str(CHECKPOINT_DIR / "encoder_supervised_best.pt"))
            torch.save(model.state_dict(), str(CHECKPOINT_DIR / "dr_grading_best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    return model.encoder


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2: encoder pretraining")
    parser.add_argument(
        "--variant", type=str, choices=["A", "B", "C"], default="C",
        help="A=SSL, B=supervised, C=SSL then supervised",
    )
    parser.add_argument("--metadata", type=str, default="data/eyepacs/metadata.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = TrainConfig(batch_size=args.batch_size, learning_rate=args.lr)
    _setup_logging(args.variant)

    # Load dataset manifest
    manifest = DatasetManifest(
        dataset_name="eyepacs",
        root_dir=EYEPACS_CONFIG.root,
    )
    manifest.load_metadata(args.metadata)
    manifest.exclude_low_quality()
    manifest.build_patient_splits()

    train_records = manifest.get_split("train")
    val_records = manifest.get_split("val")

    if args.variant in ("A", "C"):
        # SSL pretraining
        ssl_dataset = EyePACSDataset(train_records, mode="ssl")
        ssl_loader = DataLoader(
            ssl_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        )
        encoder = train_ssl(cfg, ssl_loader)
        logger.info("SSL pretraining complete.")
    else:
        encoder = None

    if args.variant in ("B", "C"):
        # Supervised DR grading
        train_ds = EyePACSDataset(train_records, mode="supervised")
        val_ds = EyePACSDataset(val_records, mode="supervised")
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=True,
        )
        encoder = train_supervised(cfg, train_loader, val_loader, encoder=encoder)
        logger.info("Supervised DR grading complete.")

    logger.info("Stage 2 finished (variant %s).", args.variant)


if __name__ == "__main__":
    main()
