"""
IDRiD dataset loader for Stage 5 lesion segmentation.

IDRiD provides per-pixel masks for four lesion types:
  microaneurysm (MA), haemorrhage (HE), hard exudate (EX), soft exudate (SE).

Each sample returns the fundus image and a (C, H, W) mask tensor where C is
the number of lesion classes.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from dr_pipeline.config import GLOBAL_VIEW_SIZE, IDRID_CONFIG, LESION_CLASSES
from dr_pipeline.datasets.preprocessing import DRPreprocessor

logger = logging.getLogger(__name__)

# IDRiD uses specific subfolder names for each lesion mask
_IDRID_MASK_DIRS = {
    "microaneurysm": "Microaneurysms",
    "haemorrhage": "Haemorrhages",
    "hard_exudate": "Hard Exudates",
    "soft_exudate": "Soft Exudates",
}


class IDRiDDataset(Dataset):
    """
    PyTorch dataset for IDRiD segmentation masks.

    Parameters
    ----------
    records : list of dict
        Must contain ``image_path`` and optionally per-lesion mask paths.
    root_dir : str | Path
        Dataset root.
    target_size : tuple of int
        Spatial size to resize both image and masks.
    preprocessor : DRPreprocessor | None
        Custom preprocessor for the image (masks are resized separately).
    is_training : bool
        Enable augmentation.
    """

    def __init__(
        self,
        records: List[Dict],
        root_dir: str = IDRID_CONFIG.root,
        target_size: Tuple[int, int] = GLOBAL_VIEW_SIZE,
        preprocessor: Optional[DRPreprocessor] = None,
        is_training: bool = True,
    ) -> None:
        self.records = records
        self.root_dir = Path(root_dir)
        self.target_size = target_size
        self.is_training = is_training

        if preprocessor is None:
            preprocessor = DRPreprocessor(
                target_size=target_size,
                is_training=is_training,
            )
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.records)

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Load a single binary mask image and return as (H, W) uint8."""
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros(self.target_size, dtype=np.uint8)
        # Binarise: anything > 0 is positive
        mask = (mask > 0).astype(np.uint8)
        mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                          interpolation=cv2.INTER_NEAREST)
        return mask

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img_path = self.root_dir / rec["image_path"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")

        # Preprocess image
        img_tensor = self.preprocessor(image)

        # Build multi-channel mask (C, H, W)
        masks = []
        for cls_name in LESION_CLASSES:
            mask_key = f"mask_{cls_name}"
            if mask_key in rec and rec[mask_key]:
                mask_path = str(self.root_dir / rec[mask_key])
                mask = self._load_mask(mask_path)
            else:
                mask = np.zeros(self.target_size, dtype=np.uint8)
            masks.append(mask)

        mask_tensor = torch.from_numpy(
            np.stack(masks, axis=0).astype(np.float32)
        )

        return img_tensor, mask_tensor
