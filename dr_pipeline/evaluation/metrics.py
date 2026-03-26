"""
Evaluation metrics for multi-lesion classification and segmentation.

Section 4.4 / 4.6 reporting requirements:
  - Per-lesion AUC and PR-AUC
  - Macro and micro averaged F1
  - Sensitivity at fixed specificity (and vice versa)
  - Dice / IoU per lesion (segmentation)
  - Evaluation by lesion-size buckets
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from dr_pipeline.config import LESION_CLASSES


# ------------------------------------------------------------------
# Classification metrics
# ------------------------------------------------------------------

def per_lesion_auc(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """
    Compute ROC-AUC per lesion class.

    Parameters
    ----------
    y_true : (N, C) binary ground truth.
    y_score : (N, C) predicted probabilities.

    Returns
    -------
    dict mapping lesion name to AUC value.
    """
    results: Dict[str, float] = {}
    for i, cls in enumerate(LESION_CLASSES):
        if y_true[:, i].sum() == 0:
            results[cls] = float("nan")
            continue
        results[cls] = float(roc_auc_score(y_true[:, i], y_score[:, i]))
    return results


def per_lesion_pr_auc(
    y_true: np.ndarray, y_score: np.ndarray
) -> Dict[str, float]:
    """Compute PR-AUC (average precision) per lesion class."""
    results: Dict[str, float] = {}
    for i, cls in enumerate(LESION_CLASSES):
        if y_true[:, i].sum() == 0:
            results[cls] = float("nan")
            continue
        results[cls] = float(average_precision_score(y_true[:, i], y_score[:, i]))
    return results


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.90,
) -> Dict[str, Tuple[float, float]]:
    """
    For each lesion, find the sensitivity at a fixed specificity.

    Returns dict mapping lesion name to (sensitivity, threshold).
    """
    results: Dict[str, Tuple[float, float]] = {}
    for i, cls in enumerate(LESION_CLASSES):
        if y_true[:, i].sum() == 0:
            results[cls] = (float("nan"), float("nan"))
            continue
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
        specificity = 1.0 - fpr
        # Find the operating point closest to target
        idx = np.argmin(np.abs(specificity - target_specificity))
        results[cls] = (float(tpr[idx]), float(thresholds[idx]))
    return results


def specificity_at_sensitivity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_sensitivity: float = 0.90,
) -> Dict[str, Tuple[float, float]]:
    """
    For each lesion, find the specificity at a fixed sensitivity.

    Returns dict mapping lesion name to (specificity, threshold).
    """
    results: Dict[str, Tuple[float, float]] = {}
    for i, cls in enumerate(LESION_CLASSES):
        if y_true[:, i].sum() == 0:
            results[cls] = (float("nan"), float("nan"))
            continue
        fpr, tpr, thresholds = roc_curve(y_true[:, i], y_score[:, i])
        idx = np.argmin(np.abs(tpr - target_sensitivity))
        specificity = 1.0 - fpr[idx]
        results[cls] = (float(specificity), float(thresholds[idx]))
    return results


def multilabel_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute per-class, macro, and micro F1 scores.

    Parameters
    ----------
    y_true : (N, C) binary ground truth.
    y_pred : (N, C) binary predictions (after thresholding).
    """
    results: Dict[str, float] = {}
    for i, cls in enumerate(LESION_CLASSES):
        results[f"f1_{cls}"] = float(
            f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        )
    results["f1_macro"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    results["f1_micro"] = float(
        f1_score(y_true, y_pred, average="micro", zero_division=0)
    )
    return results


# ------------------------------------------------------------------
# Segmentation metrics
# ------------------------------------------------------------------

def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient for two binary masks."""
    intersection = (pred * target).sum()
    return float((2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6))


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute IoU for two binary masks."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float((intersection + 1e-6) / (union + 1e-6))


def segmentation_metrics_by_size(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    size_buckets: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate segmentation stratified by lesion size.

    Section 4.6: evaluate by lesion-size buckets to avoid misleading averages.

    Parameters
    ----------
    pred_masks : (N, C, H, W) binary predictions.
    gt_masks : (N, C, H, W) binary ground truth.
    size_buckets : list of (min_pixels, max_pixels) defining buckets.

    Returns
    -------
    Nested dict: bucket_name -> metric_name -> value.
    """
    if size_buckets is None:
        size_buckets = [
            (0, 100),        # tiny (microaneurysms)
            (100, 1000),     # small
            (1000, 10000),   # medium
            (10000, np.inf), # large
        ]

    bucket_names = [f"{lo}-{hi}" for lo, hi in size_buckets]
    results: Dict[str, Dict[str, float]] = {b: {} for b in bucket_names}

    N, C = pred_masks.shape[:2]

    for c, cls_name in enumerate(LESION_CLASSES[:C]):
        for b_idx, (lo, hi) in enumerate(size_buckets):
            b_name = bucket_names[b_idx]
            dice_vals = []
            iou_vals = []

            for n in range(N):
                gt_area = gt_masks[n, c].sum()
                if lo <= gt_area < hi:
                    d = dice_score(pred_masks[n, c], gt_masks[n, c])
                    i = iou_score(pred_masks[n, c], gt_masks[n, c])
                    dice_vals.append(d)
                    iou_vals.append(i)

            if dice_vals:
                results[b_name][f"dice_{cls_name}"] = float(np.mean(dice_vals))
                results[b_name][f"iou_{cls_name}"] = float(np.mean(iou_vals))
                results[b_name][f"count_{cls_name}"] = len(dice_vals)

    return results


# ------------------------------------------------------------------
# Full evaluation report
# ------------------------------------------------------------------

def classification_report(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """
    Generate a comprehensive classification report.

    Combines AUC, PR-AUC, F1, and clinical operating points into a
    single dict suitable for JSON serialisation.
    """
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)

    y_pred = (y_score >= thresholds).astype(float)

    report = {
        "roc_auc": per_lesion_auc(y_true, y_score),
        "pr_auc": per_lesion_pr_auc(y_true, y_score),
        "f1_scores": multilabel_f1(y_true, y_pred),
        "sensitivity_at_spec90": {
            k: {"sensitivity": v[0], "threshold": v[1]}
            for k, v in sensitivity_at_specificity(y_true, y_score, 0.90).items()
        },
        "specificity_at_sens90": {
            k: {"specificity": v[0], "threshold": v[1]}
            for k, v in specificity_at_sensitivity(y_true, y_score, 0.90).items()
        },
        "optimal_thresholds": {
            LESION_CLASSES[i]: float(thresholds[i])
            for i in range(len(LESION_CLASSES))
        },
    }
    return report
