"""
Multi-Lesion Diabetic Retinopathy (DR) Staged Learning Pipeline.

A paper-grade pipeline that leverages complementary public datasets through
staged transfer learning for lesion-level DR analysis:

  Stage 2: Encoder pretraining on EyePACS (self-supervised and/or supervised)
  Stage 3: Multi-lesion classification on DDR
  Stage 4: Explainability and weak localization
  Stage 5: Lesion segmentation on IDRiD / e-ophtha

Key design principles:
  - Patient-level splits to prevent data leakage
  - Multi-scale processing for small-lesion sensitivity
  - Quantified explainability (not just qualitative)
  - Cross-dataset domain-shift awareness
"""

__version__ = "0.1.0"
