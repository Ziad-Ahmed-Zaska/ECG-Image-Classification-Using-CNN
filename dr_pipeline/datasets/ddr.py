"""
DDR (Diabetic Retinopathy Detection) dataset loader for Stage 3.

Provides multi-label lesion tags per image:
  microaneurysm, haemorrhage, hard_exudate, soft_exudate.

Supports the multi-scale branch by also returning high-res patches.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dr_pipeline.config import (
    DDR_CONFIG,
    GLOBAL_VIEW_SIZE,
    LESION_CLASSES,
    PATCH_SIZE,
    PATCHES_PER_IMAGE,
)
from dr_pipeline.datasets.preprocessing import DRPreprocessor

logger = logging.getLogger(__name__)


class DDRDataset(Dataset):
    """
    PyTorch dataset for DDR multi-label lesion classification.

    Each sample returns:
      - global_view : (C, H, W) tensor at GLOBAL_VIEW_SIZE
      - patches     : (N, C, pH, pW) tensor of high-res crops (optional)
      - labels      : (NUM_LESION_CLASSES,) float tensor of multi-hot labels

    Parameters
    ----------
    records : list of dict
        Must contain ``image_path`` and one binary column per lesion class.
    root_dir : str | Path
        Dataset root.
    return_patches : bool
        If True, also return high-res patches for the multi-scale branch.
    preprocessor : DRPreprocessor | None
        Optional custom preprocessor.
    """

    def __init__(
        self,
        records: List[Dict],
        root_dir: str = DDR_CONFIG.root,
        return_patches: bool = True,
        preprocessor: Optional[DRPreprocessor] = None,
    ) -> None:
        self.records = records
        self.root_dir = Path(root_dir)
        self.return_patches = return_patches

        if preprocessor is None:
            preprocessor = DRPreprocessor(
                target_size=GLOBAL_VIEW_SIZE,
                is_training=True,
            )
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = self.root_dir / rec["image_path"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Global view
        global_view = self.preprocessor(image)

        # Multi-hot label vector
        labels = torch.zeros(len(LESION_CLASSES), dtype=torch.float32)
        for i, cls_name in enumerate(LESION_CLASSES):
            labels[i] = float(rec.get(cls_name, 0))

        if not self.return_patches:
            return global_view, labels

        # High-res patches for multi-scale branch
        patches = self.preprocessor.extract_patches(
            image,
            patch_size=PATCH_SIZE,
            num_patches=PATCHES_PER_IMAGE,
            strategy="random",
        )
        if patches:
            patches_tensor = torch.stack(patches)
        else:
            # Fallback: single resized patch
            patches_tensor = global_view.unsqueeze(0)

        return global_view, patches_tensor, labels
