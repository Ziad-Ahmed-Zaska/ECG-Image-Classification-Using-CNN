"""
Pseudo-mask bootstrapping from DDR attribution maps (Section 4.7, optional).

Section 2.5 / 4.7 of the plan:
  - Generate weak masks from Grad-CAM attributions on DDR
  - Filter by confidence and morphology priors
  - Mix with true masks for segmentation training
  - Must be treated as an ablation with noise analysis
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from dr_pipeline.config import LESION_CLASSES
from dr_pipeline.evaluation.explainability import GradCAM


# ------------------------------------------------------------------
# Morphology priors per lesion type
# ------------------------------------------------------------------

# Expected area ranges (in pixels at 512x512) and circularity bounds
_MORPHOLOGY_PRIORS = {
    "microaneurysm": {
        "min_area": 5,
        "max_area": 500,
        "min_circularity": 0.5,
    },
    "haemorrhage": {
        "min_area": 50,
        "max_area": 20000,
        "min_circularity": 0.1,
    },
    "hard_exudate": {
        "min_area": 20,
        "max_area": 10000,
        "min_circularity": 0.2,
    },
    "soft_exudate": {
        "min_area": 100,
        "max_area": 30000,
        "min_circularity": 0.15,
    },
}


def _circularity(contour) -> float:
    """Compute circularity = 4 * pi * area / perimeter^2."""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1e-6:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


# ------------------------------------------------------------------
# Pseudo-mask generation
# ------------------------------------------------------------------

def generate_pseudo_masks(
    model: nn.Module,
    images: torch.Tensor,
    confidence_threshold: float = 0.7,
    cam_threshold: float = 0.5,
) -> List[Dict[str, np.ndarray]]:
    """
    Generate pseudo-masks from Grad-CAM heatmaps.

    Parameters
    ----------
    model : nn.Module
        Trained lesion classifier with ``encoder.layer4``.
    images : (B, C, H, W) tensor.
    confidence_threshold : float
        Only generate masks for lesions where the classifier confidence
        exceeds this threshold.
    cam_threshold : float
        Binarisation threshold for the Grad-CAM heatmap.

    Returns
    -------
    list of dicts, one per image.  Each dict maps lesion name to a
    binary (H, W) numpy mask.
    """
    device = next(model.parameters()).device
    model.eval()

    gradcam = GradCAM(model)
    B = images.size(0)
    H, W = images.shape[2], images.shape[3]

    # Get model confidences
    with torch.no_grad():
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()

    results: List[Dict[str, np.ndarray]] = []

    for i in range(B):
        masks: Dict[str, np.ndarray] = {}
        img = images[i : i + 1].to(device)

        for c, cls_name in enumerate(LESION_CLASSES):
            # Skip low-confidence predictions
            if probs[i, c] < confidence_threshold:
                continue

            # Generate Grad-CAM
            heatmap = gradcam.generate(img, c, image_size=(H, W))

            # Binarise
            binary = (heatmap >= cam_threshold).astype(np.uint8)

            # Morphology filtering
            binary = _filter_by_morphology(binary, cls_name)

            if binary.sum() > 0:
                masks[cls_name] = binary

        results.append(masks)

    return results


def _filter_by_morphology(mask: np.ndarray, lesion_type: str) -> np.ndarray:
    """
    Filter a binary mask using morphology priors for the given lesion type.

    Removes connected components that fall outside the expected area and
    circularity range.
    """
    priors = _MORPHOLOGY_PRIORS.get(lesion_type)
    if priors is None:
        return mask

    # Find connected components
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered = np.zeros_like(mask)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        circ = _circularity(cnt)

        if area < priors["min_area"] or area > priors["max_area"]:
            continue
        if circ < priors["min_circularity"]:
            continue

        cv2.drawContours(filtered, [cnt], -1, 1, cv2.FILLED)

    return filtered


# ------------------------------------------------------------------
# Noise analysis utilities (Section 4.7)
# ------------------------------------------------------------------

def pseudo_mask_quality_metrics(
    pseudo_masks: List[Dict[str, np.ndarray]],
    true_masks: List[Dict[str, np.ndarray]],
) -> Dict[str, Dict[str, float]]:
    """
    Compare pseudo-masks against true segmentation masks to quantify
    noise levels.

    Returns per-lesion precision, recall, and Dice of the pseudo-masks
    vs the ground truth.

    Use in the ablation section to report when pseudo-masks help and
    when they hurt.
    """
    results: Dict[str, Dict[str, float]] = {}

    for cls_name in LESION_CLASSES:
        tp_total = 0.0
        fp_total = 0.0
        fn_total = 0.0

        for pseudo, gt in zip(pseudo_masks, true_masks):
            p = pseudo.get(cls_name, np.zeros((1, 1)))
            g = gt.get(cls_name, np.zeros_like(p))

            # Resize if shapes differ
            if p.shape != g.shape:
                p = cv2.resize(p.astype(np.uint8), (g.shape[1], g.shape[0]))

            tp_total += (p * g).sum()
            fp_total += (p * (1 - g)).sum()
            fn_total += ((1 - p) * g).sum()

        precision = tp_total / (tp_total + fp_total + 1e-8)
        recall = tp_total / (tp_total + fn_total + 1e-8)
        dice = (2 * tp_total + 1e-8) / (2 * tp_total + fp_total + fn_total + 1e-8)

        results[cls_name] = {
            "precision": float(precision),
            "recall": float(recall),
            "dice": float(dice),
        }

    return results


def create_mixed_mask_dataset(
    true_records: List[Dict],
    pseudo_records: List[Dict],
    mixing_ratio: float = 0.5,
    seed: int = 42,
) -> List[Dict]:
    """
    Combine true and pseudo-mask records for segmentation training.

    Parameters
    ----------
    true_records : records with real pixel masks.
    pseudo_records : records with pseudo-masks from attribution maps.
    mixing_ratio : fraction of pseudo-mask records in the mix.
    seed : random seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    n_pseudo = int(len(true_records) * mixing_ratio / (1 - mixing_ratio + 1e-8))
    n_pseudo = min(n_pseudo, len(pseudo_records))

    selected_pseudo = rng.choice(
        len(pseudo_records), size=n_pseudo, replace=False
    )
    # Tag records so we can ablate later
    for r in true_records:
        r["mask_source"] = "true"
    selected = [pseudo_records[i] for i in selected_pseudo]
    for r in selected:
        r["mask_source"] = "pseudo"

    mixed = true_records + selected
    rng.shuffle(mixed)
    return mixed
