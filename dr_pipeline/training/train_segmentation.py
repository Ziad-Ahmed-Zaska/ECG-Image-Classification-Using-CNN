"""
Stage 5 training: lesion segmentation on IDRiD (and optionally e-ophtha).

Section 4.6 of the plan:
  - Reuse encoder from Stage 2 or Stage 3
  - U-Net decoder predicting per-pixel masks per lesion type
  - Dice + BCE or Focal Tversky loss
  - Evaluate by lesion-size buckets

Usage
-----
    python -m dr_pipeline.training.train_segmentation --encoder-ckpt checkpoints/encoder_ssl_best.pt
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dr_pipeline.config import (
    CHECKPOINT_DIR,
    IDRID_CONFIG,
    LESION_CLASSES,
    LOG_DIR,
    NUM_LESION_CLASSES,
    TrainConfig,
    ensure_dirs,
)
from dr_pipeline.datasets.dataset_manifest import DatasetManifest
from dr_pipeline.datasets.idrid import IDRiDDataset
from dr_pipeline.models.encoder import DREncoder
from dr_pipeline.models.segmentation import LesionSegmentationModel

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "train_segmentation.log"),
        ],
    )


# ------------------------------------------------------------------
# Metric computation
# ------------------------------------------------------------------

def compute_seg_metrics(
    pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute per-class Dice and IoU for a batch.

    Parameters
    ----------
    pred : (B, C, H, W) logit tensor.
    target : (B, C, H, W) binary mask tensor.

    Returns
    -------
    dict with keys like ``"dice_microaneurysm"``, ``"iou_haemorrhage"``, etc.
    """
    pred_bin = (torch.sigmoid(pred) >= threshold).float()
    metrics: Dict[str, float] = {}

    for c, cls_name in enumerate(LESION_CLASSES):
        p = pred_bin[:, c].contiguous().view(-1)
        t = target[:, c].contiguous().view(-1)

        tp = (p * t).sum().item()
        fp = (p * (1 - t)).sum().item()
        fn = ((1 - p) * t).sum().item()

        dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)

        metrics[f"dice_{cls_name}"] = dice
        metrics[f"iou_{cls_name}"] = iou

    # Macro averages
    metrics["dice_macro"] = np.mean(
        [metrics[f"dice_{c}"] for c in LESION_CLASSES]
    )
    metrics["iou_macro"] = np.mean(
        [metrics[f"iou_{c}"] for c in LESION_CLASSES]
    )
    return metrics


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_segmentation(
    cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder: Optional[DREncoder] = None,
) -> LesionSegmentationModel:
    """Train the segmentation model and return the best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LesionSegmentationModel(
        encoder=encoder,
        num_classes=NUM_LESION_CLASSES,
        decoder_type=cfg.seg_decoder,
        loss_type=cfg.seg_loss,
        freeze_encoder=False,
    ).to(device)

    # Use lower LR for encoder, higher for decoder
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": cfg.learning_rate * 0.1},
        {"params": decoder_params, "lr": cfg.learning_rate},
    ], weight_decay=cfg.weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.mixed_precision)

    best_val_dice = 0.0
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                pred = model(images)
                loss = model.compute_loss(pred, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_batches = 0
        all_metrics: Dict[str, list] = {
            f"{m}_{c}": [] for m in ("dice", "iou") for c in LESION_CLASSES
        }

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                pred = model(images)
                loss = model.compute_loss(pred, masks)
                val_loss += loss.item()
                val_batches += 1

                batch_metrics = compute_seg_metrics(pred, masks)
                for k, v in batch_metrics.items():
                    if k in all_metrics:
                        all_metrics[k].append(v)

        val_loss /= max(val_batches, 1)
        avg_metrics = {k: np.mean(v) for k, v in all_metrics.items() if v}
        val_dice_macro = np.mean(
            [avg_metrics.get(f"dice_{c}", 0.0) for c in LESION_CLASSES]
        )
        elapsed = time.time() - t0

        scheduler.step(val_loss)

        metrics_str = "  ".join(
            f"{c}: dice={avg_metrics.get(f'dice_{c}', 0):.3f}"
            for c in LESION_CLASSES
        )
        logger.info(
            "Seg Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  "
            "val_dice_macro=%.4f  time=%.1fs\n  %s",
            epoch, cfg.epochs, train_loss, val_loss,
            val_dice_macro, elapsed, metrics_str,
        )

        if val_dice_macro > best_val_dice:
            best_val_dice = val_dice_macro
            patience_counter = 0
            torch.save(
                model.state_dict(),
                str(CHECKPOINT_DIR / "segmentation_best.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best
    model.load_state_dict(
        torch.load(str(CHECKPOINT_DIR / "segmentation_best.pt"), map_location=device)
    )
    return model


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 5: segmentation")
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--metadata", type=str, default="data/idrid/metadata.csv")
    parser.add_argument("--loss", type=str, default="dice_bce",
                        choices=["dice_bce", "focal_tversky"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    _setup_logging()
    cfg = TrainConfig(
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seg_loss=args.loss,
    )

    encoder = None
    if args.encoder_ckpt:
        encoder = DREncoder.from_checkpoint(
            args.encoder_ckpt, backbone_name=cfg.encoder_name
        )
        logger.info("Loaded encoder from %s", args.encoder_ckpt)

    manifest = DatasetManifest(dataset_name="idrid", root_dir=IDRID_CONFIG.root)
    manifest.load_metadata(args.metadata)
    manifest.build_patient_splits()

    train_ds = IDRiDDataset(manifest.get_split("train"), is_training=True)
    val_ds = IDRiDDataset(manifest.get_split("val"), is_training=False)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model = train_segmentation(cfg, train_loader, val_loader, encoder=encoder)
    logger.info("Stage 5 segmentation training complete.")


if __name__ == "__main__":
    main()
