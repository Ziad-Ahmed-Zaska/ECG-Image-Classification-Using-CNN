"""
Multi-scale lesion classifier with attention-based MIL (Stage 3 enhancement).

Section 2.2 / 4.4 of the plan: adds a patch-level branch for small lesion
sensitivity (microaneurysms, tiny haemorrhages) and fuses it with the global
branch via gated attention.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_pipeline.config import LESION_CLASSES, NUM_LESION_CLASSES
from dr_pipeline.models.encoder import DREncoder


class GatedAttentionPooling(nn.Module):
    """
    Gated attention mechanism for Multiple Instance Learning.

    Given a bag of instance embeddings, produces a single bag-level
    representation as a weighted sum of instances.

    Reference: Ilse et al., "Attention-based Deep Multiple Instance
    Learning", ICML 2018.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.attention_V = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)

    def forward(
        self, instances: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        instances : (B, K, D) tensor of K instance embeddings.

        Returns
        -------
        bag_feat : (B, D) attention-weighted bag representation.
        attn_weights : (B, K) normalised attention scores.
        """
        v = self.attention_V(instances)  # (B, K, hidden)
        u = self.attention_U(instances)  # (B, K, hidden)
        logits = self.attention_w(v * u).squeeze(-1)  # (B, K)
        attn_weights = F.softmax(logits, dim=1)  # (B, K)
        bag_feat = torch.bmm(attn_weights.unsqueeze(1), instances).squeeze(1)
        return bag_feat, attn_weights


class MultiScaleLesionClassifier(nn.Module):
    """
    Two-branch classifier: global view + high-res patch MIL.

    Architecture
    ------------
    1. **Global branch**: encoder processes full image at GLOBAL_VIEW_SIZE.
    2. **Patch branch**: same (or shared) encoder processes K high-res
       patches; a gated attention module aggregates them.
    3. **Fusion**: concatenation of global and patch features, followed
       by a multi-label classification head.

    Parameters
    ----------
    encoder : DREncoder
        Shared backbone (weights optionally frozen for the patch branch).
    num_classes : int
        Number of lesion types.
    share_encoder : bool
        If True the same encoder processes both global and patch inputs.
        If False a separate copy is used for patches.
    loss_weights : dict | None
        Per-class positive weights for BCE loss.
    """

    def __init__(
        self,
        encoder: Optional[DREncoder] = None,
        num_classes: int = NUM_LESION_CLASSES,
        share_encoder: bool = True,
        loss_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        if encoder is None:
            encoder = DREncoder(pretrained_imagenet=True)

        self.global_encoder = encoder
        if share_encoder:
            self.patch_encoder = encoder  # shared weights
        else:
            import copy
            self.patch_encoder = copy.deepcopy(encoder)

        feat_dim = encoder.get_feat_dim()

        # Gated attention pooling for patch bag
        self.attention_pool = GatedAttentionPooling(
            in_dim=feat_dim, hidden_dim=256
        )

        # Fusion head: global_feat || patch_feat -> logits
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim * 2, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

        self.num_classes = num_classes

        if loss_weights is not None:
            w = torch.tensor(
                [loss_weights.get(c, 1.0) for c in LESION_CLASSES],
                dtype=torch.float32,
            )
        else:
            w = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("pos_weight", w)

    def forward(
        self,
        global_view: torch.Tensor,
        patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        global_view : (B, C, H, W)
        patches : (B, K, C, pH, pW)

        Returns
        -------
        logits : (B, num_classes)
        attn_weights : (B, K) patch-level attention scores.
        """
        # Global branch
        g_feat = self.global_encoder(global_view)  # (B, D)

        # Patch branch
        B, K, C, pH, pW = patches.shape
        patches_flat = patches.view(B * K, C, pH, pW)
        p_feat = self.patch_encoder(patches_flat)  # (B*K, D)
        p_feat = p_feat.view(B, K, -1)  # (B, K, D)

        bag_feat, attn_weights = self.attention_pool(p_feat)  # (B, D), (B, K)

        # Fuse
        fused = torch.cat([g_feat, bag_feat], dim=1)  # (B, 2D)
        logits = self.classifier(fused)
        return logits, attn_weights

    def compute_loss(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

    def predict_proba(
        self, global_view: torch.Tensor, patches: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            logits, _ = self.forward(global_view, patches)
            return torch.sigmoid(logits)
