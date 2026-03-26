"""
Stage 3 training: multi-lesion classification on DDR.

Section 4.4 of the plan:
  - Replace DR head with multi-label lesion head
  - Handle imbalance via weighted loss and/or balanced sampling
  - Tune lesion-specific decision thresholds on validation
  - Optional multi-scale (patch) branch for small lesions

Usage
-----
    python -m dr_pipeline.training.train_lesion_classifier --encoder-ckpt checkpoints/encoder_ssl_best.pt
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dr_pipeline.config import (
    CHECKPOINT_DIR,
    DDR_CONFIG,
    LESION_CLASSES,
    LOG_DIR,
    NUM_LESION_CLASSES,
    TrainConfig,
    ensure_dirs,
)
from dr_pipeline.datasets.dataset_manifest import DatasetManifest
from dr_pipeline.datasets.ddr import DDRDataset
from dr_pipeline.models.encoder import DREncoder
from dr_pipeline.models.lesion_classifier import MultiLesionClassifier
from dr_pipeline.models.multi_scale import MultiScaleLesionClassifier

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    ensure_dirs()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "train_lesion_classifier.log"),
        ],
    )


# ------------------------------------------------------------------
# Threshold tuning on validation set
# ------------------------------------------------------------------

def tune_thresholds(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    multi_scale: bool = False,
) -> torch.Tensor:
    """
    Sweep per-class thresholds on the validation split to maximise F1.

    Returns a (NUM_LESION_CLASSES,) tensor of optimal thresholds.
    """
    all_probs = []
    all_targets = []
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            if multi_scale:
                global_view, patches, labels = batch
                global_view = global_view.to(device)
                patches = patches.to(device)
                logits, _ = model(global_view, patches)
            else:
                images, labels = batch
                images = images.to(device)
                logits = model(images)

            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(labels)

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    best_thresholds = np.full(NUM_LESION_CLASSES, 0.5)

    for c in range(NUM_LESION_CLASSES):
        best_f1 = 0.0
        for t in np.arange(0.1, 0.9, 0.05):
            preds = (all_probs[:, c] >= t).astype(float)
            tp = ((preds == 1) & (all_targets[:, c] == 1)).sum()
            fp = ((preds == 1) & (all_targets[:, c] == 0)).sum()
            fn = ((preds == 0) & (all_targets[:, c] == 1)).sum()
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[c] = t

        logger.info(
            "Threshold for %s: %.2f  (val F1=%.4f)",
            LESION_CLASSES[c], best_thresholds[c], best_f1,
        )

    return torch.tensor(best_thresholds, dtype=torch.float32)


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------

def train_lesion_classifier(
    cfg: TrainConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    encoder: Optional[DREncoder] = None,
    multi_scale: bool = False,
) -> Tuple[nn.Module, torch.Tensor]:
    """
    Train the multi-label lesion classifier on DDR.

    Returns (model, optimal_thresholds).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if multi_scale:
        model = MultiScaleLesionClassifier(
            encoder=encoder, loss_weights=cfg.lesion_loss_weights
        ).to(device)
    else:
        model = MultiLesionClassifier(
            encoder=encoder, loss_weights=cfg.lesion_loss_weights
        ).to(device)

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
        model.train()
        train_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for batch in train_loader:
            if multi_scale:
                global_view, patches, labels = batch
                global_view = global_view.to(device)
                patches = patches.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                    logits, _ = model(global_view, patches)
                    loss = model.compute_loss(logits, labels)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=cfg.mixed_precision):
                    logits = model(images)
                    loss = model.compute_loss(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            n_samples += labels.size(0)

        train_loss /= max(n_samples, 1)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        val_n = 0

        with torch.no_grad():
            for batch in val_loader:
                if multi_scale:
                    gv, p, lbl = batch
                    gv, p, lbl = gv.to(device), p.to(device), lbl.to(device)
                    logits, _ = model(gv, p)
                else:
                    img, lbl = batch
                    img, lbl = img.to(device), lbl.to(device)
                    logits = model(img)
                loss = model.compute_loss(logits, lbl)
                val_loss += loss.item() * lbl.size(0)
                val_n += lbl.size(0)

        val_loss /= max(val_n, 1)
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        logger.info(
            "Lesion Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  time=%.1fs",
            epoch, cfg.epochs, train_loss, val_loss, elapsed,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                model.state_dict(),
                str(CHECKPOINT_DIR / "lesion_classifier_best.pt"),
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

    # Load best weights
    model.load_state_dict(
        torch.load(str(CHECKPOINT_DIR / "lesion_classifier_best.pt"), map_location=device)
    )

    # Tune thresholds on validation
    thresholds = tune_thresholds(model, val_loader, device, multi_scale=multi_scale)
    torch.save(thresholds, str(CHECKPOINT_DIR / "lesion_thresholds.pt"))

    return model, thresholds


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 3: lesion classification")
    parser.add_argument("--encoder-ckpt", type=str, default=None)
    parser.add_argument("--metadata", type=str, default="data/ddr/metadata.csv")
    parser.add_argument("--multi-scale", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    _setup_logging()
    cfg = TrainConfig(batch_size=args.batch_size, learning_rate=args.lr)

    # Load encoder from Stage 2
    encoder = None
    if args.encoder_ckpt:
        encoder = DREncoder.from_checkpoint(
            args.encoder_ckpt, backbone_name=cfg.encoder_name
        )
        logger.info("Loaded encoder from %s", args.encoder_ckpt)

    # Build DDR dataset
    manifest = DatasetManifest(dataset_name="ddr", root_dir=DDR_CONFIG.root)
    manifest.load_metadata(args.metadata)
    manifest.build_patient_splits()

    train_ds = DDRDataset(
        manifest.get_split("train"),
        return_patches=args.multi_scale,
    )
    val_ds = DDRDataset(
        manifest.get_split("val"),
        return_patches=args.multi_scale,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )

    model, thresholds = train_lesion_classifier(
        cfg, train_loader, val_loader,
        encoder=encoder, multi_scale=args.multi_scale,
    )

    logger.info("Stage 3 complete.  Optimal thresholds: %s", thresholds.tolist())


if __name__ == "__main__":
    main()
