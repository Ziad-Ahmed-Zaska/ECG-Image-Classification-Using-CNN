"""
Central configuration for the multi-lesion DR staged learning pipeline.

All hyper-parameters, paths, and dataset metadata live here so every module
draws from a single source of truth.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Lesion taxonomy (shared across stages 3-5)
# ---------------------------------------------------------------------------
LESION_CLASSES: List[str] = [
    "microaneurysm",
    "haemorrhage",
    "hard_exudate",
    "soft_exudate",
]

DR_GRADES: List[str] = [
    "no_dr",
    "mild",
    "moderate",
    "severe",
    "proliferative",
]

NUM_LESION_CLASSES: int = len(LESION_CLASSES)
NUM_DR_GRADES: int = len(DR_GRADES)


# ---------------------------------------------------------------------------
# Resolution policy (Section 4.2 of the plan)
# ---------------------------------------------------------------------------
GLOBAL_VIEW_SIZE: Tuple[int, int] = (512, 512)
PATCH_SIZE: Tuple[int, int] = (256, 256)
PATCHES_PER_IMAGE: int = 8  # for MIL / patch branch


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------
@dataclass
class DatasetConfig:
    """Metadata and paths for a single dataset."""
    name: str
    root: str
    label_type: str  # "dr_grade", "lesion_tag", "pixel_mask"
    num_classes: int
    split_file: Optional[str] = None  # CSV with patient-level splits
    quality_exclusion_file: Optional[str] = None


EYEPACS_CONFIG = DatasetConfig(
    name="eyepacs",
    root="data/eyepacs",
    label_type="dr_grade",
    num_classes=NUM_DR_GRADES,
)

DDR_CONFIG = DatasetConfig(
    name="ddr",
    root="data/ddr",
    label_type="lesion_tag",
    num_classes=NUM_LESION_CLASSES,
)

IDRID_CONFIG = DatasetConfig(
    name="idrid",
    root="data/idrid",
    label_type="pixel_mask",
    num_classes=NUM_LESION_CLASSES,
)

EOPHTHA_CONFIG = DatasetConfig(
    name="e_ophtha",
    root="data/e_ophtha",
    label_type="pixel_mask",
    num_classes=2,  # microaneurysm + exudate subsets
)


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Shared training hyper-parameters (overridable per stage)."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    epochs: int = 100
    early_stop_patience: int = 10
    num_workers: int = 4
    seed: int = 42
    mixed_precision: bool = True
    # Encoder backbone
    encoder_name: str = "resnet50"
    pretrained_imagenet: bool = True
    # Self-supervised pretraining specifics (Stage 2A)
    ssl_temperature: float = 0.07
    ssl_projection_dim: int = 128
    ssl_epochs: int = 200
    # Multi-label lesion classifier (Stage 3)
    lesion_loss_weights: Dict[str, float] = field(default_factory=lambda: {
        "microaneurysm": 3.0,
        "haemorrhage": 1.5,
        "hard_exudate": 1.0,
        "soft_exudate": 2.0,
    })
    # Segmentation (Stage 5)
    seg_decoder: str = "unet"  # "unet" or "fpn"
    seg_loss: str = "dice_bce"  # "dice_bce", "focal_tversky"


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    """Settings for evaluation, calibration, and explainability."""
    # Explainability
    attribution_method: str = "gradcam"  # "gradcam", "integrated_gradients"
    pointing_game_threshold: float = 0.5
    # Calibration
    calibration_bins: int = 15
    temperature_scaling: bool = True
    # Clinical operating points
    fixed_sensitivity: float = 0.90
    fixed_specificity: float = 0.90
    # Ensembling
    ensemble_size: int = 3


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "dr_outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIGURE_DIR = OUTPUT_DIR / "figures"


def ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    for d in (OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR):
        d.mkdir(parents=True, exist_ok=True)
