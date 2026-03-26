"""
Explainability and weak localization (Stage 4).

Section 2.3 / 4.5 of the plan:
  - Attribution maps per lesion label (Grad-CAM, Integrated Gradients)
  - Quantitative localization: pointing game accuracy, localization AUC
  - Qualitative panel generation
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_pipeline.config import LESION_CLASSES


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Produces a heatmap showing which spatial regions the model attends to
    for a given target class.

    Parameters
    ----------
    model : nn.Module
        Must have an ``encoder`` attribute with a ``layer4`` sub-module.
    target_layer : str
        Name of the layer to hook (default: ``"encoder.layer4"``).
    """

    def __init__(self, model: nn.Module, target_layer: str = "encoder.layer4"):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Register hooks on the target layer
        layer = dict(model.named_modules())[target_layer]
        layer.register_forward_hook(self._forward_hook)
        layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image: torch.Tensor,
        target_class: int,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given class.

        Parameters
        ----------
        image : (1, C, H, W) input tensor.
        target_class : int
            Index of the target lesion class.
        image_size : (H, W) or None
            Size to resize the heatmap to.  If None, uses input size.

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1].
        """
        self.model.eval()
        image = image.requires_grad_(True)

        # Forward
        output = self.model(image)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        # Backward for target class
        self.model.zero_grad()
        target_score = output[0, target_class]
        target_score.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations.")

        # Global average pool the gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Resize
        if image_size is None:
            image_size = (image.shape[2], image.shape[3])
        cam = cv2.resize(cam, (image_size[1], image_size[0]))

        return cam


class IntegratedGradients:
    """
    Integrated Gradients attribution method.

    Computes the path integral of gradients from a baseline (black image)
    to the input, attributing importance to each pixel.
    """

    def __init__(self, model: nn.Module, steps: int = 50):
        self.model = model
        self.steps = steps

    def generate(
        self,
        image: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """
        Compute integrated gradients for the given class.

        Parameters
        ----------
        image : (1, C, H, W) input tensor.
        target_class : int

        Returns
        -------
        attribution : (H, W) float32 array in [0, 1].
        """
        self.model.eval()
        baseline = torch.zeros_like(image)
        scaled_inputs = [
            baseline + (float(i) / self.steps) * (image - baseline)
            for i in range(self.steps + 1)
        ]

        grads = []
        for scaled in scaled_inputs:
            scaled = scaled.requires_grad_(True)
            output = self.model(scaled)
            if output.dim() == 1:
                output = output.unsqueeze(0)
            score = output[0, target_class]
            self.model.zero_grad()
            score.backward()
            grads.append(scaled.grad.detach().clone())

        avg_grad = torch.stack(grads).mean(dim=0)
        ig = (image - baseline) * avg_grad
        # Sum over channels for spatial attribution
        attr = ig.squeeze(0).sum(dim=0).cpu().numpy()

        # Normalise
        attr = np.abs(attr)
        attr_max = attr.max()
        if attr_max > 1e-8:
            attr = attr / attr_max

        return attr


# ------------------------------------------------------------------
# Quantitative localization (Section 2.3)
# ------------------------------------------------------------------

def pointing_game_accuracy(
    heatmaps: List[np.ndarray],
    masks: List[np.ndarray],
) -> float:
    """
    Pointing game: does the peak of the attribution fall inside the
    ground-truth mask?

    Parameters
    ----------
    heatmaps : list of (H, W) attribution maps.
    masks : list of (H, W) binary ground-truth masks.

    Returns
    -------
    accuracy : float in [0, 1].
    """
    hits = 0
    total = 0

    for hm, mask in zip(heatmaps, masks):
        if mask.sum() == 0:
            continue  # skip images with no lesion
        total += 1
        # Find peak location
        peak = np.unravel_index(hm.argmax(), hm.shape)
        if mask[peak[0], peak[1]] > 0:
            hits += 1

    return hits / max(total, 1)


def localization_auc(
    heatmaps: List[np.ndarray],
    masks: List[np.ndarray],
    num_thresholds: int = 50,
) -> float:
    """
    Localization AUC: sweep heatmap thresholds and compute the area
    under the precision-recall curve of the binarised heatmap vs the
    ground-truth mask.

    Returns
    -------
    auc_value : float.
    """
    from sklearn.metrics import auc as sk_auc

    precisions = []
    recalls = []

    all_hm = np.concatenate([h.ravel() for h in heatmaps])
    all_gt = np.concatenate([m.ravel() for m in masks])

    for t in np.linspace(0.0, 1.0, num_thresholds):
        pred = (all_hm >= t).astype(float)
        tp = (pred * all_gt).sum()
        fp = (pred * (1 - all_gt)).sum()
        fn = ((1 - pred) * all_gt).sum()

        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        precisions.append(prec)
        recalls.append(rec)

    # Sort by recall for proper AUC
    sorted_pairs = sorted(zip(recalls, precisions))
    recalls_sorted = [p[0] for p in sorted_pairs]
    precisions_sorted = [p[1] for p in sorted_pairs]

    return float(sk_auc(recalls_sorted, precisions_sorted))


# ------------------------------------------------------------------
# Attribution generation helper
# ------------------------------------------------------------------

def generate_attributions(
    model: nn.Module,
    images: torch.Tensor,
    method: str = "gradcam",
    target_classes: Optional[List[int]] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Generate attribution maps for a batch of images across lesion classes.

    Parameters
    ----------
    model : nn.Module
        Lesion classifier with ``encoder.layer4``.
    images : (B, C, H, W) tensor.
    method : ``"gradcam"`` or ``"integrated_gradients"``.
    target_classes : list of class indices.  If None, all lesion classes.

    Returns
    -------
    dict mapping lesion name to list of (H, W) heatmaps.
    """
    if target_classes is None:
        target_classes = list(range(len(LESION_CLASSES)))

    if method == "gradcam":
        explainer = GradCAM(model)
    elif method == "integrated_gradients":
        explainer = IntegratedGradients(model)
    else:
        raise ValueError(f"Unknown method: {method}")

    results: Dict[str, List[np.ndarray]] = {
        LESION_CLASSES[c]: [] for c in target_classes
    }

    for i in range(images.size(0)):
        img = images[i : i + 1]
        for c in target_classes:
            hm = explainer.generate(img, c)
            results[LESION_CLASSES[c]].append(hm)

    return results
