"""
Visualization utilities for figures and paper montages.

Section 4.10 of the plan:
  - Attribution overlay montages
  - Segmentation overlay montages
  - Failure case montages
  - Calibration reliability diagrams
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from dr_pipeline.config import FIGURE_DIR, LESION_CLASSES, ensure_dirs


def _ensure_figure_dir() -> Path:
    ensure_dirs()
    return FIGURE_DIR


# ------------------------------------------------------------------
# Heatmap overlay
# ------------------------------------------------------------------

def overlay_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a heatmap on a fundus image.

    Parameters
    ----------
    image : (H, W, 3) uint8 BGR image.
    heatmap : (H, W) float in [0, 1].
    alpha : blending weight.
    colormap : OpenCV colormap.

    Returns
    -------
    (H, W, 3) uint8 blended image.
    """
    hm_uint8 = (heatmap * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, colormap)
    hm_color = cv2.resize(hm_color, (image.shape[1], image.shape[0]))
    blended = cv2.addWeighted(image, 1 - alpha, hm_color, alpha, 0)
    return blended


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay a binary segmentation mask on an image.

    Parameters
    ----------
    image : (H, W, 3) uint8 BGR image.
    mask : (H, W) binary mask.
    color : BGR colour for the mask overlay.
    alpha : blending weight.
    """
    overlay = image.copy()
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


# ------------------------------------------------------------------
# Montage builders
# ------------------------------------------------------------------

# Lesion class -> overlay colour (BGR)
LESION_COLORS = {
    "microaneurysm": (0, 0, 255),    # red
    "haemorrhage": (0, 165, 255),     # orange
    "hard_exudate": (0, 255, 255),    # yellow
    "soft_exudate": (255, 255, 0),    # cyan
}


def build_attribution_montage(
    images: List[np.ndarray],
    heatmaps: Dict[str, List[np.ndarray]],
    max_images: int = 8,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a qualitative attribution panel.

    Rows = images, columns = original + one heatmap overlay per lesion.

    Parameters
    ----------
    images : list of (H, W, 3) BGR images.
    heatmaps : dict mapping lesion name to list of (H, W) heatmaps.
    max_images : cap on number of rows.
    save_path : if set, save the montage to this path.

    Returns
    -------
    (montage_H, montage_W, 3) uint8 image.
    """
    n = min(len(images), max_images)
    lesion_names = [c for c in LESION_CLASSES if c in heatmaps]
    cols = 1 + len(lesion_names)

    # Standardise cell size
    cell_h, cell_w = 256, 256
    montage = np.zeros((n * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i in range(n):
        img = cv2.resize(images[i], (cell_w, cell_h))
        montage[i * cell_h : (i + 1) * cell_h, 0:cell_w] = img

        for j, cls_name in enumerate(lesion_names):
            hm = heatmaps[cls_name][i]
            hm_resized = cv2.resize(hm, (cell_w, cell_h))
            blended = overlay_heatmap(img, hm_resized)
            col_start = (j + 1) * cell_w
            montage[i * cell_h : (i + 1) * cell_h, col_start : col_start + cell_w] = blended

    if save_path:
        cv2.imwrite(save_path, montage)

    return montage


def build_segmentation_montage(
    images: List[np.ndarray],
    pred_masks: List[Dict[str, np.ndarray]],
    gt_masks: Optional[List[Dict[str, np.ndarray]]] = None,
    max_images: int = 8,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Create a segmentation overlay panel.

    Columns: original | pred overlay | GT overlay (if provided).
    """
    n = min(len(images), max_images)
    has_gt = gt_masks is not None
    cols = 3 if has_gt else 2
    cell_h, cell_w = 256, 256
    montage = np.zeros((n * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i in range(n):
        img = cv2.resize(images[i], (cell_w, cell_h))
        montage[i * cell_h : (i + 1) * cell_h, 0:cell_w] = img

        # Predicted overlay
        pred_overlay = img.copy()
        for cls_name, mask in pred_masks[i].items():
            mask_resized = cv2.resize(mask.astype(np.uint8), (cell_w, cell_h))
            color = LESION_COLORS.get(cls_name, (0, 255, 0))
            pred_overlay = overlay_mask(pred_overlay, mask_resized, color=color)
        montage[i * cell_h : (i + 1) * cell_h, cell_w : 2 * cell_w] = pred_overlay

        # GT overlay
        if has_gt and gt_masks[i]:
            gt_overlay = img.copy()
            for cls_name, mask in gt_masks[i].items():
                mask_resized = cv2.resize(mask.astype(np.uint8), (cell_w, cell_h))
                color = LESION_COLORS.get(cls_name, (0, 255, 0))
                gt_overlay = overlay_mask(gt_overlay, mask_resized, color=color)
            montage[i * cell_h : (i + 1) * cell_h, 2 * cell_w : 3 * cell_w] = gt_overlay

    if save_path:
        cv2.imwrite(save_path, montage)

    return montage


def build_failure_montage(
    images: List[np.ndarray],
    captions: List[str],
    heatmaps: Optional[List[np.ndarray]] = None,
    max_images: int = 8,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Failure case montage with short captions.

    Section 4.9 / 4.10: vessel confusion, bright artifacts, missed MAs.
    """
    n = min(len(images), max_images)
    cell_h, cell_w = 256, 256
    caption_h = 40
    total_cell_h = cell_h + caption_h
    cols = 2 if heatmaps else 1
    montage = np.ones((n * total_cell_h, cols * cell_w, 3), dtype=np.uint8) * 255

    for i in range(n):
        img = cv2.resize(images[i], (cell_w, cell_h))
        y_start = i * total_cell_h
        montage[y_start : y_start + cell_h, 0:cell_w] = img

        if heatmaps:
            hm = cv2.resize(heatmaps[i], (cell_w, cell_h))
            blended = overlay_heatmap(img, hm)
            montage[y_start : y_start + cell_h, cell_w : 2 * cell_w] = blended

        # Add caption text
        caption = captions[i][:50]  # truncate long captions
        cv2.putText(
            montage, caption,
            (5, y_start + cell_h + 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
        )

    if save_path:
        cv2.imwrite(save_path, montage)

    return montage


# ------------------------------------------------------------------
# Calibration reliability diagram
# ------------------------------------------------------------------

def plot_reliability_diagram(
    bin_accuracies: np.ndarray,
    bin_confidences: np.ndarray,
    bin_counts: np.ndarray,
    ece: float,
    save_path: Optional[str] = None,
) -> np.ndarray:
    """
    Draw a reliability diagram (calibration plot) using OpenCV.

    Returns the diagram as a (H, W, 3) uint8 image.
    This avoids a matplotlib dependency for the core pipeline.
    """
    H, W = 400, 400
    margin = 50
    plot_w = W - 2 * margin
    plot_h = H - 2 * margin
    canvas = np.ones((H, W, 3), dtype=np.uint8) * 255

    n_bins = len(bin_accuracies)
    bar_w = plot_w // n_bins

    # Draw bars
    for i in range(n_bins):
        if bin_counts[i] == 0:
            continue
        x = margin + i * bar_w
        bar_height = int(bin_accuracies[i] * plot_h)
        y_top = margin + plot_h - bar_height
        cv2.rectangle(canvas, (x, y_top), (x + bar_w - 2, margin + plot_h),
                       (200, 120, 50), -1)

    # Draw diagonal (perfect calibration)
    cv2.line(canvas,
             (margin, margin + plot_h),
             (margin + plot_w, margin),
             (0, 0, 0), 1)

    # Axes
    cv2.line(canvas, (margin, margin), (margin, margin + plot_h), (0, 0, 0), 2)
    cv2.line(canvas, (margin, margin + plot_h),
             (margin + plot_w, margin + plot_h), (0, 0, 0), 2)

    # Labels
    cv2.putText(canvas, f"ECE={ece:.3f}", (margin + 5, margin + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    cv2.putText(canvas, "Confidence", (W // 2 - 30, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if save_path:
        cv2.imwrite(save_path, canvas)

    return canvas
