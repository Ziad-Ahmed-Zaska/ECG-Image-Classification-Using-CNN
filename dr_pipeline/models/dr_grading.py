"""
Supervised DR grading head (Stage 2, Variant B).

Fine-tunes the encoder on EyePACS 5-class DR severity labels.  Can also
serve as Variant C when initialised from a self-supervised checkpoint.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_pipeline.config import NUM_DR_GRADES
from dr_pipeline.models.encoder import DREncoder


class DRGradingModel(nn.Module):
    """
    Encoder + linear DR grading head.

    Parameters
    ----------
    encoder : DREncoder | None
        Pre-initialised encoder.  If None a fresh one is created.
    num_classes : int
        Number of DR severity grades (default 5).
    dropout : float
        Dropout before the final linear layer.
    """

    def __init__(
        self,
        encoder: Optional[DREncoder] = None,
        num_classes: int = NUM_DR_GRADES,
        dropout: float = 0.3,
    ):
        super().__init__()
        if encoder is None:
            encoder = DREncoder(pretrained_imagenet=True)
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder.get_feat_dim(), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, num_classes) logits."""
        feat = self.encoder(x)
        return self.head(feat)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated softmax probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)

    def save_encoder(self, path: str) -> None:
        """Persist the encoder weights for downstream stages."""
        torch.save(
            {"encoder_state_dict": self.encoder.state_dict()},
            path,
        )
