"""
Multi-label lesion classifier (Stage 3).

Replaces the DR grading head with a multi-label lesion head trained on DDR.
Handles class imbalance via weighted BCE and optional balanced sampling.
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_pipeline.config import LESION_CLASSES, NUM_LESION_CLASSES
from dr_pipeline.models.encoder import DREncoder


class MultiLesionClassifier(nn.Module):
    """
    Encoder + multi-label sigmoid head for lesion classification.

    Parameters
    ----------
    encoder : DREncoder | None
        Pre-trained encoder (from Stage 2).
    num_classes : int
        Number of lesion types.
    dropout : float
        Dropout rate.
    loss_weights : dict | None
        Per-class positive weights for BCEWithLogitsLoss.
    """

    def __init__(
        self,
        encoder: Optional[DREncoder] = None,
        num_classes: int = NUM_LESION_CLASSES,
        dropout: float = 0.3,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if encoder is None:
            encoder = DREncoder(pretrained_imagenet=True)
        self.encoder = encoder
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.get_feat_dim(), 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        # Per-class positive weights for imbalance handling
        if loss_weights is not None:
            w = torch.tensor(
                [loss_weights.get(c, 1.0) for c in LESION_CLASSES],
                dtype=torch.float32,
            )
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("pos_weight", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, num_classes) raw logits."""
        feat = self.encoder(x)
        return self.head(feat)

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Weighted binary cross-entropy with logits."""
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Sigmoid probabilities per lesion."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)

    def predict(
        self, x: torch.Tensor, thresholds: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Hard predictions using per-class thresholds.

        Parameters
        ----------
        thresholds : (num_classes,) tensor or None
            If None, 0.5 is used for all classes.
        """
        proba = self.predict_proba(x)
        if thresholds is None:
            thresholds = torch.full(
                (self.num_classes,), 0.5, device=proba.device
            )
        return (proba >= thresholds).float()
