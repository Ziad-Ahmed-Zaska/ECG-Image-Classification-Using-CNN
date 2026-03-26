"""
EyePACS dataset loader for Stage 2 encoder pretraining.

Supports both:
  - Supervised DR grading (Variant B)
  - Self-supervised image-only loading (Variant A)

Uses patient-level splits managed by DatasetManifest.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from dr_pipeline.config import DR_GRADES, EYEPACS_CONFIG, GLOBAL_VIEW_SIZE
from dr_pipeline.datasets.preprocessing import DRPreprocessor

logger = logging.getLogger(__name__)


class EyePACSDataset(Dataset):
    """
    PyTorch dataset for EyePACS fundus images.

    Parameters
    ----------
    records : list of dict
        Image-level records from DatasetManifest (must contain at least
        ``image_path`` and ``label`` keys).
    root_dir : str | Path
        Root directory where images are stored.
    mode : str
        ``"supervised"`` for DR grading labels, ``"ssl"`` for two-view
        self-supervised pairs.
    preprocessor : DRPreprocessor | None
        Custom preprocessor instance.  If None a default one is built.
    ssl_transforms : tuple of Callable | None
        (view1_transform, view2_transform) used when mode="ssl".
    """

    def __init__(
        self,
        records: List[Dict],
        root_dir: str = EYEPACS_CONFIG.root,
        mode: str = "supervised",
        preprocessor: Optional[DRPreprocessor] = None,
        ssl_transforms: Optional[Tuple[Callable, Callable]] = None,
    ) -> None:
        self.records = records
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.grade_to_idx = {g: i for i, g in enumerate(DR_GRADES)}

        if preprocessor is None:
            preprocessor = DRPreprocessor(
                target_size=GLOBAL_VIEW_SIZE,
                is_training=(mode != "test"),
            )
        self.preprocessor = preprocessor

        if mode == "ssl" and ssl_transforms is None:
            self.ssl_transforms = preprocessor.build_ssl_augmentation()
        else:
            self.ssl_transforms = ssl_transforms

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, idx: int) -> np.ndarray:
        rec = self.records[idx]
        img_path = self.root_dir / rec["image_path"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        return image

    def __getitem__(self, idx: int):
        image = self._load_image(idx)
        rec = self.records[idx]

        if self.mode == "ssl":
            # FOV crop + colour constancy first
            cx, cy, r = DRPreprocessor.detect_fov(image)
            image = DRPreprocessor.crop_fov(image, cx, cy, r)
            image = DRPreprocessor.grey_world(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            from PIL import Image as PILImage
            pil_img = PILImage.fromarray(image)
            view1 = self.ssl_transforms[0](pil_img)
            view2 = self.ssl_transforms[1](pil_img)
            return view1, view2

        # Supervised mode
        tensor = self.preprocessor(image)
        label = self.grade_to_idx.get(rec.get("label", "no_dr"), 0)
        return tensor, label
