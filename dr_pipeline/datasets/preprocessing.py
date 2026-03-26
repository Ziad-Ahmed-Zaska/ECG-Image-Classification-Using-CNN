"""
Preprocessing standardization for retinal fundus images.

Section 4.2 of the plan:
  - Field-of-view (FOV) detection and background removal
  - Colour constancy (documented method)
  - Resolution policy: global view + high-res patches
  - Clinically plausible augmentations
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torchvision import transforms


class DRPreprocessor:
    """
    Deterministic preprocessing pipeline for fundus images.

    Applies (in order):
      1. FOV circle detection and background crop
      2. Colour constancy via modified grey-world
      3. Resize to the requested resolution
      4. Optional augmentation (training only)

    Parameters
    ----------
    target_size : tuple of int
        (H, W) for the output image.
    is_training : bool
        Enable stochastic augmentations when True.
    colour_constancy : str
        Method name: ``"grey_world"`` or ``"none"``.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        is_training: bool = False,
        colour_constancy: str = "grey_world",
    ) -> None:
        self.target_size = target_size
        self.is_training = is_training
        self.colour_constancy = colour_constancy

    # ------------------------------------------------------------------
    # FOV detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_fov(image: np.ndarray) -> Tuple[int, int, int]:
        """
        Detect the circular field-of-view in a fundus image.

        Returns (cx, cy, radius).  Falls back to image centre if
        detection fails.
        """
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 15, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            h, w = image.shape[:2]
            return w // 2, h // 2, min(w, h) // 2

        largest = max(contours, key=cv2.contourArea)
        (cx, cy), radius = cv2.minEnclosingCircle(largest)
        return int(cx), int(cy), int(radius)

    @staticmethod
    def crop_fov(
        image: np.ndarray, cx: int, cy: int, radius: int, padding: int = 10
    ) -> np.ndarray:
        """Crop image to the bounding box of the FOV circle."""
        h, w = image.shape[:2]
        r = radius + padding
        x1 = max(cx - r, 0)
        y1 = max(cy - r, 0)
        x2 = min(cx + r, w)
        y2 = min(cy + r, h)
        return image[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Colour constancy
    # ------------------------------------------------------------------

    @staticmethod
    def grey_world(image: np.ndarray) -> np.ndarray:
        """
        Modified grey-world colour constancy.

        Normalises each channel so that its mean equals the global mean
        intensity.  Keeps pixel values in [0, 255].
        """
        img = image.astype(np.float32)
        means = img.mean(axis=(0, 1))
        global_mean = means.mean()
        # Avoid division by zero
        scales = np.where(means > 1e-6, global_mean / means, 1.0)
        img *= scales[np.newaxis, np.newaxis, :]
        return np.clip(img, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Augmentation transforms (torchvision)
    # ------------------------------------------------------------------

    def _build_augmentation(self) -> transforms.Compose:
        """
        Clinically plausible augmentations (Section 4.2).

        * Geometric: mild rotation (+/-15 deg), horizontal/vertical flip.
        * Photometric: brightness/contrast jitter within small range.
        * No aggressive spatial warps that distort lesion morphology.
        """
        aug_list: list = []
        if self.is_training:
            aug_list += [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02
                ),
            ]
        aug_list += [
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(aug_list)

    def build_ssl_augmentation(self) -> Tuple[transforms.Compose, transforms.Compose]:
        """
        Two-view augmentation pair for self-supervised pretraining (SimCLR).

        Each view applies independent random crop, colour jitter, blur,
        and flip so the network learns invariance.
        """
        shared = [
            transforms.RandomResizedCrop(self.target_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
        return transforms.Compose(shared), transforms.Compose(shared)

    # ------------------------------------------------------------------
    # Full preprocessing call
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        Run the full preprocessing pipeline on a raw BGR fundus image.

        Parameters
        ----------
        image : np.ndarray
            BGR uint8 image as read by cv2.imread.

        Returns
        -------
        torch.Tensor
            Preprocessed (C, H, W) float tensor.
        """
        # 1. FOV crop
        cx, cy, r = self.detect_fov(image)
        image = self.crop_fov(image, cx, cy, r)

        # 2. Colour constancy
        if self.colour_constancy == "grey_world":
            image = self.grey_world(image)

        # 3. Convert BGR -> RGB for torchvision
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 4. Augmentation + resize + normalise via torchvision
        from PIL import Image as PILImage

        pil_img = PILImage.fromarray(image)
        transform = self._build_augmentation()
        return transform(pil_img)

    # ------------------------------------------------------------------
    # Patch extraction for multi-scale branch
    # ------------------------------------------------------------------

    def extract_patches(
        self,
        image: np.ndarray,
        patch_size: Tuple[int, int] = (256, 256),
        num_patches: int = 8,
        strategy: str = "random",
    ) -> List[torch.Tensor]:
        """
        Extract high-resolution patches from a preprocessed image.

        Parameters
        ----------
        image : np.ndarray
            Full-resolution fundus image (BGR).
        patch_size : tuple
            (H, W) of each patch.
        num_patches : int
            Number of patches to extract.
        strategy : str
            ``"random"`` or ``"grid"``.

        Returns
        -------
        list of torch.Tensor
            Each tensor is (C, pH, pW).
        """
        # FOV crop first
        cx, cy, r = self.detect_fov(image)
        image = self.crop_fov(image, cx, cy, r)
        if self.colour_constancy == "grey_world":
            image = self.grey_world(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        ph, pw = patch_size
        patches: List[torch.Tensor] = []

        patch_transform = transforms.Compose([
            transforms.Resize(patch_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        from PIL import Image as PILImage

        if strategy == "random":
            for _ in range(num_patches):
                y = np.random.randint(0, max(h - ph, 1))
                x = np.random.randint(0, max(w - pw, 1))
                crop = image[y : y + ph, x : x + pw]
                pil_crop = PILImage.fromarray(crop)
                patches.append(patch_transform(pil_crop))
        elif strategy == "grid":
            step_y = max((h - ph) // int(np.sqrt(num_patches)), 1)
            step_x = max((w - pw) // int(np.sqrt(num_patches)), 1)
            for y in range(0, h - ph + 1, step_y):
                for x in range(0, w - pw + 1, step_x):
                    if len(patches) >= num_patches:
                        break
                    crop = image[y : y + ph, x : x + pw]
                    pil_crop = PILImage.fromarray(crop)
                    patches.append(patch_transform(pil_crop))
                if len(patches) >= num_patches:
                    break

        return patches
