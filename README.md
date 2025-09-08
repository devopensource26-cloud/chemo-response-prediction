# chemo-response-prediction

**Predicting chemotherapy response from gene-expression — reproducible pipeline and artifacts**

This repository contains code, notebooks, and model artifacts for a project that predicts chemotherapy response from gene expression. The project implements a full reproducible pipeline: preprocessing, priority-weighted elastic-net + stability selection, nested CV (AUCPR optimization), an *asymmetric gray-zone* detection, a gray-boosted MLP, stacking (meta-LR) and optional isotonic calibration. 
---

## Table of contents

* [Repository layout](#repository-layout)
* [Purpose and high-level design](#purpose-and-high-level-design)
* [Requirements](#requirements)
* [Quick start — reproduce pipeline](#quick-start---reproduce-pipeline)
* [Detailed folders and files](#detailed-folders-and-files)
* [Expected outputs & artifacts](#expected-outputs--artifacts)
* [Usage examples (CLI)](#usage-examples-cli)
* [Good practices and notes](#good-practices-and-notes)

---

# Repository layout

```
chemo-response-prediction/
├── data/
│   ├── raw/                    # original raw CSVs (do not modify)
│   └── processed/              # cleaned CSVs used by training (e.g. df_priority6_with_response_index.csv)
├── models/                     # saved model artifacts (MLP saved_models, lr_fold_models.joblib, stacking_meta.joblib, oof CSVs)
├── results/                    # figures and result tables (ROC/PR/calibration/DCA)
├── scripts/                    # runnable scripts (train pipeline, predict external)
├── src/                        # reusable modules (preprocessing, feature_selection, mlp, modeling, stacking, utils)
├── notebooks/                  # (optional) exploratory notebooks
├── requirements.txt
└── README.md
```

> The repository as provided uses `data/processed/df_priority6_with_response_index.csv` in examples. Adjust `--input` paths as needed.

---

# Purpose and high-level design

This project aims to build a robust, reproducible classifier of chemotherapy response from gene expression matrices. The main ideas implemented:

1. **Preprocessing**: variance filtering and removal of highly correlated features (e.g. |r| > 0.98) to reduce redundancy.
2. **Feature selection**: priority-weighted elastic-net plus repeated subsampling (stability selection).
3. **Modeling**:

   * Baseline logistic regression trained with nested cross-validation (outer folds for honest evaluation, inner folds for hyperparameter selection). AUCPR is the primary optimization metric.
   * Per-fold thresholds are selected by maximizing F1 on validation folds.
4. **Asymmetric gray zone**: detect an asymmetric ambiguous probability region around the LR threshold using quantiles of FP/FN distances to threshold. Samples inside that zone are re-scored by a dedicated MLP.
5. **Gray-boosted MLP**: MLP trained with adaptive sample weights (boosting ambiguous samples, adding extra weight to positives inside the gray region).
6. **Stacking and calibration**: build meta-features `Z = [oof_lr, oof_mlp, |oof_lr - thr|, oof_mlp - oof_lr, mask_gray]`, train a meta-logistic, and optionally calibrate probabilities using isotonic regression.

The pipeline produces OOF predictions, fold models, final stacking model and calibration object to enable honest evaluation and external inference (when preprocessing artifacts are preserved).

---

# Requirements

Minimum recommended environment:

* Python 3.8+
* Packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow` (if training MLP), `joblib`, `matplotlib`, `tqdm`

Install dependencies:

```bash
python -m venv .venv
# Linux / mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```
---

# Quick start — reproduce pipeline

1. Prepare your CSV with features and a column named `Response`. Place it in `data/processed/`, for example:

   ```
   data/processed/df_priority6_with_response_index.csv
   ```

2. Run the training script (example):

   ```bash
   python scripts/train_stacking_gray.py --input data/processed/df_priority6_with_response_index.csv --outdir models/
   ```

3. After training, check:

   * `models/` for saved models and OOF CSVs.
   * `results/` for PR/ROC/calibration/DCA plots and final metric CSVs.

---

# Detailed folders and files

### `data/`

* `raw/` — original unsanitized CSV files (keep for provenance).
* `processed/` — processed CSVs used by training; these should include a `Response` column and use the same column ordering as during training.

### `src/` (core modules)

* `preprocessing.py` — load and clean, variance thresholding, correlation pruning.
* `feature_selection.py` — stability selection using elastic-net (optionally priority-weighted).
* `modeling.py` — nested CV orchestration for baseline models.
* `mlp.py` — MLP building and training utilities (supports sample weights).
* `stacking.py` — build meta-features and train meta-model + isotonic calibration.
* `utils.py` — helpers: threshold selection, ECE, metrics, save/load helpers.

### `scripts/`

* `train_stacking_gray.py` (or `train_pipeline.py`) — orchestrator: preprocess → feature selection → nested CV LR → gray zone detection → MLP gray-boosted → stacking → save artifacts & diagnostics.
* `predict_external.py` — skeleton to apply saved artifacts to external cohorts (must be adapted to load preprocessing artifacts: feature list, scaler, vt mask, and fold models).

### `models/`

* `mlp_fold_{k}.keras` — saved MLP per fold.
* `lr_fold_models.joblib` — list/ensemble of logistic regressions (fold-level).
* `stacking_meta.joblib` — meta-model and calibration object (and recommended threshold).
* `oof_all.csv` — OOF probabilities for LR, MLP and stacked predictions together with `y`.

### `results/`

* Figures (ROC/PR with bootstrap bands, calibration plots, DCA) and CSV summaries (global metrics and per-subgroup / zone-gray metrics).

---

# Expected outputs & artifacts

After successful training you should find the following artifacts (filenames are suggestions used in the codebase):

* `models/oof_all.csv` — columns: `oof_lr`, `oof_mlp`, `oof_stack`, `y`.
* `models/lr_fold_models.joblib` — serialized list of trained logistic regressions.
* `models/mlp_fold_1.keras`, `models/mlp_fold_2.keras`, ... — MLP saved models.
* `models/stacking_meta.joblib` — contains: `meta_model`, `isotonic_calibrator` (optional), `thr_lr` (recommended threshold), and possibly `selected_features` and `scaler`.
* `results/*.png` and `results/*.csv` — diagnostic figures and summary tables.

**Important**: For inference on external datasets you must save and use the same preprocessing artifacts (feature list, scaler, variance-threshold mask). If these are not saved, exact reproduction of predictions is not guaranteed.

---

# Usage examples (CLI)

Train pipeline:

```bash
python scripts/train_stacking_gray.py \
  --input data/processed/df_priority6_with_response_index.csv \
  --outdir models/ \
  --n_splits 5 \
  --epochs 200
```

Predict on external data (skeleton):

```bash
python scripts/predict_external.py \
  --model models/stacking_meta.joblib \
  --input data/external/GSE25065_filtered.csv \
  --outdir results/
```

> NOTE: `predict_external.py` must be adapted to load `selected_features.txt` and `scaler.pkl` saved during training. The script is a template demonstrating the flow: load artifacts → apply identical preprocessing → compute LR and MLP probabilities → build `Z` → meta-model predict → optional isotonic transform.

---

# Good practices and notes

* **Save preprocessing artifacts**: always persist `selected_features.txt`, `scaler.pkl`, `variance_threshold_mask.npy`. These are required to do exact external inference.
* **Separation of concerns**: keep training and inference code separate. Training scripts should save artifacts; inference scripts should only load and apply them.
* **Calibration**: if you perform isotonic calibration, do not calibrate on the final held-out test where you report final metrics. Use a dedicated calibration split or external cohort.
* **Document thresholds**: keep recommended thresholds and gray-zone bounds in a JSON or joblib object alongside the saved models.
* **Version data**: keep `data/raw/` immutable and store processed inputs in `data/processed/` with versioned filenames.

---