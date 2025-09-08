# chemo-response-prediction

**Predicting chemotherapy response from gene-expression — reproducible pipeline and artifacts**

This repository contains code, notebooks, and model artifacts for a project that predicts chemotherapy response from gene expression. The project implements a full reproducible pipeline: preprocessing, priority-weighted elastic-net + stability selection, nested CV (AUCPR optimization), an *asymmetric gray-zone* detection, a gray-boosted MLP, stacking (meta-LR) and optional isotonic calibration.

---

## Table of contents

* [Repository layout](#repository-layout)
* [Data (hosted)](#data-hosted)
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
├── data/                       # NOT included in this GitHub repo (see Data section)
├── models/                     # saved model artifacts (MLP saved_models, lr_fold_models.joblib, stacking_meta.joblib, oof CSVs)
├── results/                    # figures and result tables (ROC/PR/calibration/DCA)
├── scripts/                    # runnable scripts (train pipeline, predict external)
├── src/                        # reusable modules (preprocessing, feature_selection, mlp, modeling, stacking, utils)
├── notebooks/                  # (optional) exploratory notebooks
├── requirements.txt
└── README.md
```

---

# Data (hosted)

The `data/` folder (both `data/raw/` and `data/processed/`) is hosted outside GitHub. Download the data files from the Hugging Face repository:

**Hugging Face (data):** `https://huggingface.co/datasets/DevOpensource/chemo-response-data/tree/main`

The Hugging Face repo contains the full `data/raw/` and `data/processed/` folders used by the project. Place the downloaded `data/` folder at the repository root before running any scripts.

---

# Purpose and high-level design

This project builds a robust, reproducible classifier of chemotherapy response from gene expression matrices. Main components:

1. **Preprocessing** — variance filtering and removal of highly correlated features (e.g. |r| > 0.98).
2. **Feature selection** — priority-weighted elastic-net + repeated subsampling (stability selection).
3. **Modeling**:

   * Baseline logistic regression trained with nested cross-validation (outer folds for honest evaluation, inner folds for hyperparameter selection). AUCPR is the primary optimization metric.
   * Per-fold thresholds are selected by maximizing F1 on validation folds.
4. **Asymmetric gray zone** — detect an asymmetric ambiguous probability region around the LR threshold using quantiles of FP/FN distances to the threshold. Samples in that zone are re-scored by an MLP.
5. **Gray-boosted MLP** — MLP trained with adaptive sample weights that boost ambiguous samples and give extra weight to positives in the gray region.
6. **Stacking and calibration** — meta-features `Z = [oof_lr, oof_mlp, |oof_lr - thr|, oof_mlp - oof_lr, mask_gray]` are used to train a meta-logistic; isotonic calibration is optional.

The pipeline produces OOF predictions, fold models, a final stacking model and a calibration object to enable honest evaluation and external inference (when preprocessing artifacts are preserved).

---

# Requirements

Minimum recommended environment:

* Python 3.8+
* Packages: `numpy`, `pandas`, `scikit-learn`, `tensorflow` (if training MLP), `joblib`, `matplotlib`, `tqdm`, `huggingface-hub` (optional, for data download)

Install dependencies:

```bash
python -m venv .venv
# Linux / mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
# optional (to download data programmatically)
pip install huggingface-hub
```

---

# Quick start — reproduce pipeline

1. Download the `data/` folder from the Hugging Face repo (`data/raw/` and `data/processed/`) and place it at the repository root.

2. Ensure your CSV includes a column named `Response`. Example used by scripts:

```
data/processed/df_priority6_with_response_index.csv
```

3. Run the training script (example):

```bash
python scripts/train_stacking_gray.py --input data/processed/df_priority6_with_response_index.csv --outdir models/
```

4. After training, check:

* `models/` for saved models and OOF CSVs.
* `results/` for PR/ROC/calibration/DCA plots and final metric CSVs.

---

# Detailed folders and files

### `data/`

* `raw/` — original raw CSVs (hosted on Hugging Face).
* `processed/` — cleaned CSVs used by training; must include a `Response` column and keep the same column ordering as in training.

### `src/` (core modules)

* `preprocessing.py` — loading and cleaning, variance thresholding, correlation pruning.
* `feature_selection.py` — stability selection using elastic-net (optionally priority-weighted).
* `modeling.py` — nested CV orchestration for baseline models.
* `mlp.py` — MLP building and training utilities (supports `sample_weight`).
* `stacking.py` — build meta-features and train meta-model + isotonic calibration.
* `utils.py` — helpers: threshold selection, ECE, metrics, save/load helpers.

### `scripts/`

* `train_stacking_gray.py` (or `train_pipeline.py`) — orchestrator: preprocess → feature selection → nested CV LR → gray zone detection → MLP gray-boosted → stacking → save artifacts & diagnostics.
* `predict_external.py` — skeleton to apply saved artifacts to external cohorts (adapt to load `selected_features.txt`, `scaler.pkl`, variance mask and fold models).

### `models/`

* `mlp_fold_{k}.keras` — saved MLP per fold.
* `lr_fold_models.joblib` — list/ensemble of logistic regressions (fold-level).
* `stacking_meta.joblib` — meta-model and calibration object (and recommended threshold).
* `oof_all.csv` — OOF probabilities for LR, MLP and stacked predictions with `y`.

### `results/`

* Figures (ROC/PR with bootstrap bands, calibration plots, DCA) and CSV summaries (global and zone-gray metrics).

---

# Expected outputs & artifacts

After training you should find:

* `models/oof_all.csv` — `oof_lr`, `oof_mlp`, `oof_stack`, `y`.
* `models/lr_fold_models.joblib` — serialized LR models by fold.
* `models/mlp_fold_1.keras`, `models/mlp_fold_2.keras`, ... — MLP saved models.
* `models/stacking_meta.joblib` — contains: `meta_model`, isotonic calibrator (optional), `thr_lr`, and optionally `selected_features` and `scaler`.
* `results/*.png` and `results/*.csv` — diagnostic figures and summary tables.

**Important**: for external inference you must save and reuse preprocessing artifacts (`selected_features.txt`, `scaler.pkl`, `variance_threshold_mask.npy`). Without them, exact reproduction of predictions is not guaranteed.

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

> NOTE: `predict_external.py` is a template and must be adapted to load `selected_features.txt` and `scaler.pkl` generated during training. Flow: load artifacts → apply identical preprocessing → compute LR & MLP probabilities → build `Z` → meta-model predict → optional isotonic transform.

---

# Good practices and notes

* Keep large data off GitHub and store the `data/` folder on Hugging Face (link above).
* Always persist preprocessing artifacts: `selected_features.txt`, `scaler.pkl`, `variance_threshold_mask.npy`. These are required for exact external inference.
* Separate training and inference code. Training scripts must save artifacts; inference scripts must only load and apply them.
* If you use isotonic calibration, do not calibrate on the final test set used to report metrics. Use a separate calibration split or an external cohort.
* Document recommended thresholds and gray-zone bounds alongside saved models (JSON or joblib).
* Keep `data/raw/` immutable and version processed inputs in `data/processed/`.

---