"""
Lesion segmentation model (Stage 5).

Reuses the encoder from Stage 2/3 and adds a U-Net-style decoder to
predict per-pixel lesion masks on IDRiD / e-ophtha.

Section 4.6 of the plan: uses losses robust to class imbalance and
small objects (Dice + BCE combo or Focal Tversky).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dr_pipeline.config import NUM_LESION_CLASSES
from dr_pipeline.models.encoder import DREncoder


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """Single U-Net decoder block: upsample + concat skip + double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch // 2 + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear",
                              align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    U-Net decoder that consumes multi-scale features from DREncoder.

    Expects feature maps keyed as ``"s1"`` .. ``"s4"`` from
    :meth:`DREncoder.forward_features`.
    """

    def __init__(self, encoder_name: str = "resnet50", num_classes: int = 4):
        super().__init__()
        # Channel dimensions per ResNet variant
        if encoder_name in ("resnet50", "resnet101"):
            ch = [2048, 1024, 512, 256]
        else:
            ch = [512, 256, 128, 64]

        self.dec4 = DecoderBlock(ch[0], ch[1], ch[1])
        self.dec3 = DecoderBlock(ch[1], ch[2], ch[2])
        self.dec2 = DecoderBlock(ch[2], ch[3], ch[3])
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(ch[3], ch[3] // 2, kernel_size=2, stride=2),
            nn.Conv2d(ch[3] // 2, ch[3] // 2, 3, padding=1),
            nn.BatchNorm2d(ch[3] // 2),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv2d(ch[3] // 2, num_classes, 1)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.dec4(features["s4"], features["s3"])
        x = self.dec3(x, features["s2"])
        x = self.dec2(x, features["s1"])
        x = self.dec1(x)
        return self.final(x)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss, computed per-class then averaged."""

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        B, C = pred.shape[:2]
        pred_flat = pred.view(B, C, -1)
        target_flat = target.view(B, C, -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Combined Dice + BCE loss for segmentation."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = self.dice(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss -- emphasises small/hard-to-segment objects.

    Tversky index generalises Dice by weighting FP and FN differently.
    The focal exponent further penalises low-confidence predictions.
    """

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        B, C = pred.shape[:2]
        pred_flat = pred.view(B, C, -1)
        target_flat = target.view(B, C, -1)

        tp = (pred_flat * target_flat).sum(dim=2)
        fp = ((1 - target_flat) * pred_flat).sum(dim=2)
        fn = (target_flat * (1 - pred_flat)).sum(dim=2)

        tversky = (tp + 1e-6) / (tp + self.alpha * fn + self.beta * fp + 1e-6)
        focal_tversky = (1.0 - tversky) ** self.gamma
        return focal_tversky.mean()


# ---------------------------------------------------------------------------
# Full segmentation model
# ---------------------------------------------------------------------------

class LesionSegmentationModel(nn.Module):
    """
    Encoder-decoder segmentation model for per-pixel lesion prediction.

    Parameters
    ----------
    encoder : DREncoder | None
        Pre-trained encoder from a previous stage.
    num_classes : int
        Number of lesion mask channels.
    decoder_type : str
        ``"unet"`` (only option currently).
    loss_type : str
        ``"dice_bce"`` or ``"focal_tversky"``.
    freeze_encoder : bool
        Freeze encoder weights during segmentation training.
    """

    def __init__(
        self,
        encoder: Optional[DREncoder] = None,
        num_classes: int = NUM_LESION_CLASSES,
        decoder_type: str = "unet",
        loss_type: str = "dice_bce",
        freeze_encoder: bool = False,
    ):
        super().__init__()
        if encoder is None:
            encoder = DREncoder(pretrained_imagenet=True)
        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Detect encoder variant from feature dimension
        encoder_name = "resnet50" if encoder.get_feat_dim() >= 2048 else "resnet18"
        self.decoder = UNetDecoder(
            encoder_name=encoder_name, num_classes=num_classes
        )

        if loss_type == "dice_bce":
            self.criterion = DiceBCELoss()
        elif loss_type == "focal_tversky":
            self.criterion = FocalTverskyLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, C, H, W) logit masks."""
        features = self.encoder.forward_features(x)
        return self.decoder(features)

    def compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # Ensure spatial sizes match
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(
                pred, size=target.shape[2:], mode="bilinear",
                align_corners=False,
            )
        return self.criterion(pred, target)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary masks."""
        with torch.no_grad():
            logits = self.forward(x)
            return (torch.sigmoid(logits) >= threshold).float()
