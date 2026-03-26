"""
Self-supervised pretraining via SimCLR (Stage 2, Variant A).

Learns general retinal anatomy features from unlabelled EyePACS images
before any task-specific supervision is applied.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_pipeline.models.encoder import DREncoder


class ProjectionHead(nn.Module):
    """
    MLP projection head used during contrastive pretraining.

    Maps encoder features to a lower-dimensional space where the NT-Xent
    loss is computed.  Discarded after pretraining.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 2048, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NTXentLoss(nn.Module):
    """
    Normalised temperature-scaled cross-entropy loss (NT-Xent).

    Given a batch of 2N views (N pairs), treats matching pairs as
    positives and all other 2(N-1) views as negatives.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z1, z2 : (N, D) projected features from the two augmented views.
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        sim = torch.mm(z, z.t()) / self.temperature  # (2N, 2N)

        # Mask out self-similarity
        mask = torch.eye(2 * N, device=sim.device, dtype=torch.bool)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+N) and (i+N, i)
        pos_idx = torch.arange(2 * N, device=sim.device)
        pos_idx = torch.cat([pos_idx[N:], pos_idx[:N]])

        labels = pos_idx
        loss = F.cross_entropy(sim, labels)
        return loss


class SimCLRPretrainer(nn.Module):
    """
    Full SimCLR module: encoder + projection head + loss.

    After pretraining, call :meth:`get_encoder` to extract the encoder
    with learned weights (projection head is discarded).

    Parameters
    ----------
    encoder : DREncoder
        Backbone encoder.
    projection_dim : int
        Output dimension of the projection head.
    temperature : float
        NT-Xent temperature.
    """

    def __init__(
        self,
        encoder: Optional[DREncoder] = None,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ):
        super().__init__()
        if encoder is None:
            encoder = DREncoder(pretrained_imagenet=True)
        self.encoder = encoder
        self.projector = ProjectionHead(
            in_dim=encoder.get_feat_dim(),
            out_dim=projection_dim,
        )
        self.criterion = NTXentLoss(temperature=temperature)

    def forward(
        self, view1: torch.Tensor, view2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        view1, view2 : (B, C, H, W) augmented image pairs.

        Returns
        -------
        loss : scalar
        z1, z2 : projected features (useful for monitoring).
        """
        h1 = self.encoder(view1)
        h2 = self.encoder(view2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        loss = self.criterion(z1, z2)
        return loss, z1, z2

    def get_encoder(self) -> DREncoder:
        """Return the encoder with learned weights (projection head dropped)."""
        return self.encoder

    def save_encoder(self, path: str) -> None:
        """Persist only the encoder state dict for downstream stages."""
        torch.save(
            {"encoder_state_dict": self.encoder.state_dict()},
            path,
        )
