"""
Calibration, uncertainty, and clinical operating points.

Section 2.4 of the plan:
  - Temperature scaling for post-hoc calibration
  - Expected Calibration Error (ECE)
  - Ensemble prediction aggregation
  - Coverage vs risk curves for selective prediction
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dr_pipeline.config import LESION_CLASSES


# ------------------------------------------------------------------
# Temperature scaling
# ------------------------------------------------------------------

class TemperatureScaling(nn.Module):
    """
    Learned temperature parameter for post-hoc calibration.

    Optimised on the validation set after training is complete.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def fit(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        max_iter: int = 100,
        lr: float = 0.01,
    ) -> float:
        """
        Optimise temperature on validation logits to minimise NLL.

        Parameters
        ----------
        logits : (N, C) raw model logits.
        targets : (N, C) binary targets (multi-label).

        Returns
        -------
        Optimal temperature value.
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        return float(self.temperature.item())


# ------------------------------------------------------------------
# Expected Calibration Error
# ------------------------------------------------------------------

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ECE across all classes jointly.

    Parameters
    ----------
    y_true : (N, C) binary labels.
    y_prob : (N, C) predicted probabilities.
    n_bins : number of confidence bins.

    Returns
    -------
    ece : float
    bin_accuracies : (n_bins,)
    bin_confidences : (n_bins,)
    bin_counts : (n_bins,)
    """
    # Flatten to treat each (sample, class) pair independently
    y_true_flat = y_true.ravel()
    y_prob_flat = y_prob.ravel()

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (y_prob_flat > lo) & (y_prob_flat <= hi)
        if mask.sum() == 0:
            continue
        bin_counts[i] = mask.sum()
        bin_accuracies[i] = y_true_flat[mask].mean()
        bin_confidences[i] = y_prob_flat[mask].mean()

    total = bin_counts.sum()
    if total == 0:
        return 0.0, bin_accuracies, bin_confidences, bin_counts

    ece = float(
        (bin_counts * np.abs(bin_accuracies - bin_confidences)).sum() / total
    )
    return ece, bin_accuracies, bin_confidences, bin_counts


# ------------------------------------------------------------------
# Ensemble predictions
# ------------------------------------------------------------------

class EnsemblePredictor:
    """
    Aggregate predictions from multiple model checkpoints.

    Improves both discrimination and calibration by averaging
    probabilities across independently trained models.
    """

    def __init__(self, models: List[nn.Module]):
        self.models = models

    def predict_proba(
        self, images: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        Average sigmoid probabilities across all ensemble members.

        Parameters
        ----------
        images : (B, C, H, W)

        Returns
        -------
        (B, num_classes) averaged probabilities.
        """
        all_probs = []
        for model in self.models:
            model.eval()
            model.to(device)
            with torch.no_grad():
                logits = model(images.to(device))
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())

        return torch.stack(all_probs).mean(dim=0)

    def predict_with_uncertainty(
        self, images: torch.Tensor, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return mean probabilities and predictive uncertainty (std).

        Useful for selective prediction: abstain when uncertainty is high.
        """
        all_probs = []
        for model in self.models:
            model.eval()
            model.to(device)
            with torch.no_grad():
                logits = model(images.to(device))
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu())

        stacked = torch.stack(all_probs)
        mean = stacked.mean(dim=0)
        std = stacked.std(dim=0)
        return mean, std


# ------------------------------------------------------------------
# Coverage vs Risk curve
# ------------------------------------------------------------------

def coverage_risk_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty: np.ndarray,
    n_points: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the coverage vs risk curve for selective prediction.

    Samples are sorted by uncertainty; at each coverage level we report
    the error rate among the most confident samples.

    Parameters
    ----------
    y_true : (N,) or (N, C) binary labels.
    y_prob : (N,) or (N, C) predicted probabilities.
    uncertainty : (N,) per-sample uncertainty (e.g., max std across classes).
    n_points : number of coverage levels.

    Returns
    -------
    coverages : (n_points,) fraction of data retained.
    risks : (n_points,) error rate at each coverage level.
    """
    # Flatten if multi-label
    if y_true.ndim > 1:
        # Per-sample error = fraction of wrong labels
        preds = (y_prob >= 0.5).astype(float)
        errors = (preds != y_true).any(axis=1).astype(float)
    else:
        errors = (np.round(y_prob) != y_true).astype(float)

    # Sort by ascending uncertainty (most confident first)
    order = np.argsort(uncertainty)
    errors_sorted = errors[order]

    N = len(errors)
    coverages = np.linspace(1.0 / N, 1.0, n_points)
    risks = np.zeros(n_points)

    for i, cov in enumerate(coverages):
        n_keep = max(int(cov * N), 1)
        risks[i] = errors_sorted[:n_keep].mean()

    return coverages, risks
