"""
e-ophtha dataset loader for external segmentation validation.

e-ophtha provides two subsets:
  - e_optha_MA : microaneurysm masks
  - e_optha_EX : exudate masks

Used primarily as an external test set for cross-dataset evaluation
(Section 4.1 of the plan).
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dr_pipeline.config import EOPHTHA_CONFIG, GLOBAL_VIEW_SIZE
from dr_pipeline.datasets.preprocessing import DRPreprocessor

logger = logging.getLogger(__name__)

# e-ophtha only covers two lesion types
EOPHTHA_CLASSES = ["microaneurysm", "hard_exudate"]


class EOPhthaDataset(Dataset):
    """
    PyTorch dataset for e-ophtha segmentation masks.

    Parameters
    ----------
    records : list of dict
        Must contain ``image_path``, ``mask_path``, and ``lesion_type``.
    root_dir : str | Path
        Dataset root.
    target_size : tuple of int
        Spatial size for both image and mask.
    preprocessor : DRPreprocessor | None
        Custom preprocessor.
    """

    def __init__(
        self,
        records: List[Dict],
        root_dir: str = EOPHTHA_CONFIG.root,
        target_size: Tuple[int, int] = GLOBAL_VIEW_SIZE,
        preprocessor: Optional[DRPreprocessor] = None,
    ) -> None:
        self.records = records
        self.root_dir = Path(root_dir)
        self.target_size = target_size

        if preprocessor is None:
            preprocessor = DRPreprocessor(
                target_size=target_size,
                is_training=False,  # typically used for external test
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

        img_tensor = self.preprocessor(image)

        # Build 2-channel mask (MA, EX)
        masks = []
        for cls_name in EOPHTHA_CLASSES:
            mask_key = f"mask_{cls_name}"
            if mask_key in rec and rec[mask_key]:
                mask_path = str(self.root_dir / rec[mask_key])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 0).astype(np.uint8)
                    mask = cv2.resize(
                        mask,
                        (self.target_size[1], self.target_size[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                else:
                    mask = np.zeros(self.target_size, dtype=np.uint8)
            else:
                mask = np.zeros(self.target_size, dtype=np.uint8)
            masks.append(mask)

        mask_tensor = torch.from_numpy(
            np.stack(masks, axis=0).astype(np.float32)
        )

        return img_tensor, mask_tensor
