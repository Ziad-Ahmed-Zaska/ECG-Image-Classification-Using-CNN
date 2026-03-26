"""
Dataset manifest and patient-level split management.

Section 4.1 of the plan: enforce patient-level splits so both eyes of the
same patient always land in the same fold.  Produces reproducible
train / val / test CSVs and a summary manifest for paper appendices.
"""

import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SplitStats:
    """Summary statistics for one data split."""
    split: str
    num_patients: int
    num_images: int
    label_distribution: Dict[str, int]


@dataclass
class DatasetManifest:
    """
    Builds and stores a reproducible dataset manifest.

    Responsibilities
    ----------------
    * Enumerate images and map them to patient IDs.
    * Exclude images that fail a quality check.
    * Produce patient-level train / val / test splits (no leakage).
    * Write the manifest as a JSON + per-split CSV for auditing.

    Parameters
    ----------
    dataset_name : str
        Human-readable dataset identifier (e.g. "eyepacs", "ddr").
    root_dir : str | Path
        Path to the dataset root on disk.
    label_column : str
        Column name that holds the target label in the source metadata.
    patient_id_column : str
        Column name that holds the patient / subject identifier.
    seed : int
        Random seed for split generation.
    train_ratio : float
        Fraction of patients assigned to training.
    val_ratio : float
        Fraction of patients assigned to validation.
    """

    dataset_name: str
    root_dir: Path
    label_column: str = "label"
    patient_id_column: str = "patient_id"
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    # Internal state
    _records: List[Dict] = field(default_factory=list, repr=False)
    _splits: Dict[str, List[Dict]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        assert self.train_ratio + self.val_ratio < 1.0, (
            "train_ratio + val_ratio must be < 1.0"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_metadata(self, metadata_path: str) -> "DatasetManifest":
        """
        Read a CSV of image-level metadata and populate internal records.

        Expected columns (at minimum): image_path, <patient_id_column>,
        <label_column>.  Additional columns are preserved.
        """
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self._records.append(row)

        logger.info(
            "Loaded %d image records from %s", len(self._records), metadata_path
        )
        return self

    def exclude_low_quality(
        self, quality_column: str = "quality_score", threshold: float = 0.3
    ) -> "DatasetManifest":
        """
        Remove images below a quality threshold.

        If the quality column does not exist in the records the step is
        silently skipped (not every dataset ships quality scores).
        """
        if not self._records:
            return self

        if quality_column not in self._records[0]:
            logger.warning(
                "Quality column '%s' not found; skipping exclusion.", quality_column
            )
            return self

        before = len(self._records)
        self._records = [
            r for r in self._records
            if float(r.get(quality_column, 1.0)) >= threshold
        ]
        logger.info(
            "Quality filter: %d -> %d images (excluded %d)",
            before, len(self._records), before - len(self._records),
        )
        return self

    def build_patient_splits(self) -> "DatasetManifest":
        """
        Assign every *patient* to train / val / test, then propagate to
        images.  Uses deterministic hashing so the split is reproducible
        without needing to persist state.
        """
        patient_to_records: Dict[str, List[Dict]] = defaultdict(list)
        for rec in self._records:
            pid = rec[self.patient_id_column]
            patient_to_records[pid].append(rec)

        patients = sorted(patient_to_records.keys())
        rng = np.random.RandomState(self.seed)
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_pids = set(patients[:n_train])
        val_pids = set(patients[n_train : n_train + n_val])
        test_pids = set(patients[n_train + n_val :])

        self._splits = {"train": [], "val": [], "test": []}
        for pid, recs in patient_to_records.items():
            if pid in train_pids:
                split = "train"
            elif pid in val_pids:
                split = "val"
            else:
                split = "test"
            for r in recs:
                r["split"] = split
                self._splits[split].append(r)

        for s in ("train", "val", "test"):
            logger.info("Split '%s': %d images", s, len(self._splits[s]))

        return self

    def get_split(self, split: str) -> List[Dict]:
        """Return records for a given split."""
        if split not in self._splits:
            raise ValueError(
                f"Split '{split}' not available.  Call build_patient_splits first."
            )
        return self._splits[split]

    def summary(self) -> List[SplitStats]:
        """Compute per-split summary statistics."""
        stats = []
        for split_name, records in self._splits.items():
            pids = {r[self.patient_id_column] for r in records}
            label_dist = dict(Counter(r[self.label_column] for r in records))
            stats.append(
                SplitStats(
                    split=split_name,
                    num_patients=len(pids),
                    num_images=len(records),
                    label_distribution=label_dist,
                )
            )
        return stats

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str) -> None:
        """
        Write the manifest as:
          - manifest.json  (summary + config)
          - train.csv, val.csv, test.csv
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Summary JSON
        summary_data = {
            "dataset": self.dataset_name,
            "seed": self.seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "splits": {},
        }
        for s in self.summary():
            summary_data["splits"][s.split] = {
                "num_patients": s.num_patients,
                "num_images": s.num_images,
                "label_distribution": s.label_distribution,
            }
        with open(out / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2)

        # Per-split CSVs
        for split_name, records in self._splits.items():
            if not records:
                continue
            csv_path = out / f"{split_name}.csv"
            fieldnames = list(records[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(records)

        logger.info("Manifest saved to %s", out)

    @classmethod
    def load(cls, manifest_dir: str) -> "DatasetManifest":
        """Reload a previously saved manifest from its JSON + CSVs."""
        manifest_dir = Path(manifest_dir)
        with open(manifest_dir / "manifest.json", encoding="utf-8") as f:
            meta = json.load(f)

        obj = cls(
            dataset_name=meta["dataset"],
            root_dir=str(manifest_dir),
            seed=meta["seed"],
            train_ratio=meta["train_ratio"],
            val_ratio=meta["val_ratio"],
        )
        for split_name in ("train", "val", "test"):
            csv_path = manifest_dir / f"{split_name}.csv"
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    obj._splits[split_name] = list(reader)
        return obj
