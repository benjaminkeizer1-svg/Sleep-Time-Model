# Sleep Stage Classification from Wrist Actigraphy

Patient-calibrated machine learning models for automated sleep staging using exclusively wrist-worn accelerometer data, developed for BME1580.

## Overview

This project rigorously quantifies what wrist accelerometry can and cannot distinguish in sleep staging. Wrist-worn accelerometers capture gross motor activity but lack the neurophysiological signals (EEG, EOG, EMG) that clinically define sleep stages. Rather than claiming clinical-grade accuracy from a fundamentally limited modality, we establish principled performance baselines using proper subject-level cross-validation and extensive feature engineering, characterizing which stage distinctions are recoverable from motion data alone.

Two classification granularities and two model architectures are explored:

| Model | Stages | Description |
|-------|--------|-------------|
| **3-Stage Random Forest** | Wake, NREM, REM | Coarse classification with calibrated RF |
| **3-Stage XGBoost** | Wake, NREM, REM | Coarse classification with gradient boosting |
| **5-Stage Random Forest** | Wake, N1, N2, N3, REM | Full clinical staging with calibrated RF |
| **5-Stage XGBoost** | Wake, N1, N2, N3, REM | Full clinical staging with gradient boosting |

## Dataset

Uses the [Motion and Heart Rate from a Wrist-Worn Wearable and Labeled Sleep from Polysomnography](https://physionet.org/content/sleep-accel/1.0.0/) dataset from PhysioNet. Only the accelerometer data is used — heart rate / PPG signals are intentionally excluded to assess the limits of motion-only sleep staging.

## Methodology

### Feature Engineering (44 features)
- **Base epoch features**: mean, std, max, and range of vector magnitude (VM); tilt angle statistics
- **Spectral features**: spectral entropy, dominant frequency, power in low/high bands, power ratios, zero-crossing rate
- **Inter-axis correlations**: pairwise correlations between x, y, z axes
- **Multi-scale rolling windows**: 2-min, 5-min, 15-min, and 30-min rolling mean/std of VM
- **Temporal features**: time-of-night, VM delta (1st and 2nd order), time since last movement
- **Subject-normalized z-scores**: within-subject normalization of key motion features
- **Contextual features**: VM percentile rank, lagged epochs (±1, ±2, ±3), stillness run length

### Model Pipeline
- **Cross-validation**: GroupKFold (5 splits) with subject-level grouping — no data leakage between subjects
- **Class balancing**: `class_weight='balanced'` for implicit inverse-frequency reweighting
- **Probability calibration**: CalibratedClassifierCV with isotonic calibration
- **Post-processing**: Multi-class threshold optimization (REM, N3), rolling mode smoothing, and minimum bout enforcement (3 min) to remove clinically implausible stage transitions

All reported metrics are **out-of-fold** predictions, reflecting genuine generalization performance.

**Known limitation**: Threshold optimization (REM/N3) is performed on the same OOF predictions used for evaluation, which introduces mild optimistic bias. A fully rigorous approach would use nested CV; for this course project, the limitation is acknowledged.

## Repository Structure

```
├── Patient_Calibrated_3_Stage_Random_Forest (1).ipynb
├── Patient_Calibrated_3_Stage_XGBoost (1).ipynb
├── Patient_Calibrated_5_Stage_Random_Forest (1).ipynb
├── Patient_Calibrated_5_Stage_XGBoost (1).ipynb
├── apply_improvements.py
└── .gitignore
```

## Requirements

- Python 3.8+
- NumPy, Pandas, Scikit-learn, XGBoost, imbalanced-learn
- Matplotlib, Seaborn (visualization)
- SciPy

## Usage

1. Download the dataset from [PhysioNet](https://physionet.org/content/sleep-accel/1.0.0/) and place `heartratedata.zip` in the project root.
2. Open any notebook and run all cells. The first cell handles extraction automatically.
3. Each notebook is self-contained and produces full evaluation metrics, confusion matrices, feature importances, per-subject breakdowns, and hypnogram visualizations.
