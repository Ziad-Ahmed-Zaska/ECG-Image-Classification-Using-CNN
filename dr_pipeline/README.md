# Multi-Lesion Diabetic Retinopathy Staged Learning Pipeline

A paper-grade pipeline for lesion-level diabetic retinopathy analysis that leverages complementary public datasets through staged transfer learning.

## Pipeline overview

```
EyePACS images --> Stage 2: encoder pretrain --> Stage 3: multi-lesion classifier --> Stage 4: weak localization
                                            \                                  |
                                             --> Stage 5: lesion segmentation <-+
DDR lesion labels -------> Stage 3                                              |
IDRiD / e-ophtha masks -----------------> Stage 5 <-----------------------------+
```

### Stages

| Stage | Task | Dataset | Output |
|-------|------|---------|--------|
| 2A | Self-supervised pretraining (SimCLR) | EyePACS | Encoder weights |
| 2B | Supervised DR grading | EyePACS | Encoder weights + DR head |
| 2C | SSL then supervised fine-tune | EyePACS | Encoder weights |
| 3 | Multi-label lesion classification | DDR | Per-lesion probabilities |
| 4 | Explainability / weak localization | DDR + masks | Attribution maps + localization metrics |
| 5 | Pixel-level lesion segmentation | IDRiD, e-ophtha | Per-pixel masks |
| (opt) | Pseudo-mask bootstrapping | DDR attributions | Augmented segmentation data |

### Lesion types

- Microaneurysm (MA)
- Haemorrhage (HE)
- Hard exudate (EX)
- Soft exudate (SE)

## Project structure

```
dr_pipeline/
  config.py                          # Central configuration
  datasets/
    dataset_manifest.py              # Patient-level split management
    preprocessing.py                 # FOV detection, colour constancy, augmentation
    eyepacs.py                       # EyePACS loader (Stage 2)
    ddr.py                           # DDR loader (Stage 3)
    idrid.py                         # IDRiD loader (Stage 5)
    e_ophtha.py                      # e-ophtha loader (external validation)
  models/
    encoder.py                       # Shared ResNet encoder backbone
    self_supervised.py               # SimCLR pretrainer (Stage 2A)
    dr_grading.py                    # DR grading head (Stage 2B)
    lesion_classifier.py             # Multi-label lesion head (Stage 3)
    multi_scale.py                   # Global + patch MIL branch (Stage 3+)
    segmentation.py                  # U-Net decoder + losses (Stage 5)
  training/
    train_pretrain.py                # Stage 2 training script
    train_lesion_classifier.py       # Stage 3 training script
    train_segmentation.py            # Stage 5 training script
  evaluation/
    metrics.py                       # AUC, PR-AUC, F1, Dice, IoU
    explainability.py                # Grad-CAM, Integrated Gradients, pointing game
    calibration.py                   # Temperature scaling, ECE, ensembles
  utils/
    visualization.py                 # Montage builders, overlays
    pseudo_masks.py                  # Pseudo-mask generation and filtering
```

## Quick start

### 1. Install dependencies

```bash
pip install -r dr_pipeline/requirements.txt
```

### 2. Prepare datasets

Place datasets under `data/` with the following structure:

```
data/
  eyepacs/
    images/
    metadata.csv        # columns: image_path, patient_id, label, [quality_score]
  ddr/
    images/
    metadata.csv        # columns: image_path, patient_id, microaneurysm, haemorrhage, hard_exudate, soft_exudate
  idrid/
    images/
    masks/
    metadata.csv        # columns: image_path, patient_id, mask_microaneurysm, mask_haemorrhage, ...
  e_ophtha/
    images/
    masks/
    metadata.csv
```

### 3. Run training stages

```bash
# Stage 2: Encoder pretraining (Variant C = SSL + supervised)
python -m dr_pipeline.training.train_pretrain --variant C --metadata data/eyepacs/metadata.csv

# Stage 3: Multi-lesion classification on DDR
python -m dr_pipeline.training.train_lesion_classifier \
    --encoder-ckpt dr_outputs/checkpoints/encoder_supervised_best.pt \
    --metadata data/ddr/metadata.csv \
    --multi-scale

# Stage 5: Lesion segmentation on IDRiD
python -m dr_pipeline.training.train_segmentation \
    --encoder-ckpt dr_outputs/checkpoints/encoder_supervised_best.pt \
    --metadata data/idrid/metadata.csv \
    --loss dice_bce
```

## Key design decisions

### Data integrity

- **Patient-level splits**: both eyes of the same patient are always in the same fold, preventing data leakage (the most common fatal flaw in medical imaging papers).
- **Quality filtering**: images below a configurable quality threshold are excluded before splitting.
- **Fixed test sets**: test splits are never used for threshold tuning; thresholds are optimised on the validation split only.

### Small-lesion sensitivity

- **Multi-scale architecture**: a global branch processes the full image while a patch branch extracts high-resolution crops. A gated attention mechanism (MIL) aggregates patch-level features, addressing the well-known failure mode of missing microaneurysms in global-only classifiers.

### Explainability

- **Quantified, not just qualitative**: Grad-CAM and Integrated Gradients produce attribution maps that are evaluated with pointing game accuracy and localization AUC against ground-truth masks.

### Calibration and clinical utility

- Post-hoc temperature scaling for probability calibration.
- Ensemble prediction for improved discrimination and uncertainty estimation.
- Coverage-vs-risk curves for selective prediction (reject uncertain cases).

## Ablation checklist (Section 4.8)

The following ablations form the minimum set for paper-grade evaluation:

- [ ] No pretraining (random init)
- [ ] Supervised EyePACS pretraining only (Variant B)
- [ ] Self-supervised EyePACS pretraining only (Variant A)
- [ ] SSL + supervised pretraining (Variant C)
- [ ] Full pipeline with segmentation
- [ ] With vs without multi-scale patch branch
- [ ] With vs without pseudo-mask bootstrapping
- [ ] With vs without calibration and ensembles

## Known failure modes (Section 4.9)

These should be documented with specific examples in the paper:

- Vessel segments confused with haemorrhages
- Bright imaging artifacts confused with hard exudates
- Missed microaneurysms in low-contrast peripheral regions
- Domain shift across camera types and clinical sites

## License

This project is for research purposes.
