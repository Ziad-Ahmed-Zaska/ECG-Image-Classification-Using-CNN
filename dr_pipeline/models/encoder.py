"""
Shared encoder backbone for all pipeline stages.

Wraps a torchvision ResNet (or similar) and exposes feature maps at
multiple scales so downstream heads (classification, segmentation) can
tap into the level they need.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torchvision.models as tv_models


# Map of supported backbones to their torchvision constructors and
# expected feature dimensions at the final conv layer.
_BACKBONE_REGISTRY = {
    "resnet18": (tv_models.resnet18, 512),
    "resnet34": (tv_models.resnet34, 512),
    "resnet50": (tv_models.resnet50, 2048),
    "resnet101": (tv_models.resnet101, 2048),
}


class DREncoder(nn.Module):
    """
    Fundus image encoder that returns both a feature vector and
    multi-scale feature maps.

    Parameters
    ----------
    backbone_name : str
        One of ``"resnet18"``, ``"resnet34"``, ``"resnet50"``,
        ``"resnet101"``.
    pretrained_imagenet : bool
        Initialise with ImageNet weights.
    freeze_bn : bool
        Freeze batch-norm running stats (useful when fine-tuning with
        small batches).
    """

    def __init__(
        self,
        backbone_name: str = "resnet50",
        pretrained_imagenet: bool = True,
        freeze_bn: bool = False,
    ) -> None:
        super().__init__()
        if backbone_name not in _BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from {list(_BACKBONE_REGISTRY)}"
            )

        factory, self.feat_dim = _BACKBONE_REGISTRY[backbone_name]
        weights = "IMAGENET1K_V1" if pretrained_imagenet else None
        base = factory(weights=weights)

        # Split into stages for multi-scale access
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool)
        self.layer1 = base.layer1  # stride 4
        self.layer2 = base.layer2  # stride 8
        self.layer3 = base.layer3  # stride 16
        self.layer4 = base.layer4  # stride 32

        self.avgpool = base.avgpool
        self.freeze_bn = freeze_bn

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return a dict of feature maps at each stage.

        Keys: ``"s1"`` .. ``"s4"`` plus ``"pool"`` (global avg-pool vector).
        """
        s0 = self.stem(x)
        s1 = self.layer1(s0)
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        pool = self.avgpool(s4).flatten(1)

        return {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "pool": pool}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the global average-pooled feature vector (B, feat_dim)."""
        return self.forward_features(x)["pool"]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> "DREncoder":
        """Override to keep BN frozen when requested."""
        super().train(mode)
        if self.freeze_bn and mode:
            for m in self.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        return self

    def get_feat_dim(self) -> int:
        """Return the dimensionality of the pooled feature vector."""
        return self.feat_dim

    @classmethod
    def from_checkpoint(cls, path: str, **kwargs) -> "DREncoder":
        """
        Load an encoder whose weights were saved during a previous stage.

        Typical use: load Stage 2 encoder weights before Stage 3.
        """
        encoder = cls(pretrained_imagenet=False, **kwargs)
        state = torch.load(path, map_location="cpu")
        # Handle wrapped state dicts (e.g. from SimCLR)
        if "encoder_state_dict" in state:
            state = state["encoder_state_dict"]
        encoder.load_state_dict(state, strict=False)
        return encoder
