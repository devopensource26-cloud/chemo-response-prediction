#!/usr/bin/env python3
"""
nmf_experiments.py - Optimized pipeline: run grid of NMF experiments -> train XGBoost -> compare.

Requisitos:
pip install pandas numpy scikit-learn xgboost matplotlib joblib tqdm

Ejemplo (pequeña rejilla, 5 folds):
python scripts/nmf_experiments.py --file ./data/raw/data_GSE205568_quantile_corrected_Biopsy.csv --ks 5 10 --neg_modes clip shift --w_modes relative --scales none minmax --n_splits 5 --n_repeats 2 --n_jobs 1 --max_experiments 50 --outdir results/experiments
"""
import os
import argparse
import itertools
import time
import json
from pathlib import Path
from datetime import datetime
from functools import partial
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve, average_precision_score,
                             confusion_matrix)
from joblib import Parallel, delayed, dump as joblib_dump
import matplotlib.pyplot as plt
from tqdm import tqdm

# Optional: XGBoost
try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost is required: pip install xgboost") from e

# Suppress NMF convergence warnings cluttering output (we still log them)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Grid experiments: NMF -> XGBoost (optimized)")
    p.add_argument("--file", "-f", required=True, help="CSV input file")
    p.add_argument("--ks", nargs="+", type=int, default=[5, 10], help="List of n_components (K)")
    p.add_argument("--neg_modes", nargs="+", choices=["shift", "clip", "error"],
                   default=["clip"], help="Cómo manejar valores negativos")
    p.add_argument("--w_modes", nargs="+", choices=["raw", "relative"],
                   default=["relative"], help="Qué usar como features para XGBoost")
    p.add_argument("--scales", nargs="+", choices=["none", "minmax", "log1p"], default=["none"],
                   help="Transformaciones sobre X antes NMF")
    p.add_argument("--n_splits", type=int, default=5, help="Folds para CV")
    p.add_argument("--n_repeats", type=int, default=3, help="Repeticiones para CV")
    p.add_argument("--xgb_params", type=str, default=None,
                   help="JSON string con parámetros para XGBClassifier, p.ej. '{\"n_estimators\":200}'")
    p.add_argument("--outdir", "-o", default="results/experiments", help="Directorio para resultados")
    p.add_argument("--n_jobs", type=int, default=1, help="Paralelismo para experimentos (joblib)")
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--max_iter", type=int, default=2000, help="max_iter para NMF (aumentado por defecto)")
    p.add_argument("--solver", type=str, choices=["cd", "mu"], default="cd", help="solver para NMF ('cd' o 'mu')")
    p.add_argument("--tol", type=float, default=1e-4, help="tol para NMF")
    p.add_argument("--max_experiments", type=int, default=200, help="Abortar si grid > este número (safety)")
    p.add_argument("--save_models", action="store_true", help="Guardar modelos XGBoost completos por experimento (usa espacio)")
    return p.parse_args()


# ---------- Utilities ----------
def safe_read_csv(path):
    df = pd.read_csv(path, index_col=False)
    return df


def remove_nan_inf_rows(X_df, sample_accession, response_series):
    finite_mask = np.isfinite(X_df.values).all(axis=1)
    n_before = X_df.shape[0]
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        X_df = X_df.loc[finite_mask].reset_index(drop=True)
        sample_accession = sample_accession.loc[finite_mask].reset_index(drop=True)
        response_series = response_series.loc[finite_mask].reset_index(drop=True)
    return X_df, sample_accession, response_series, n_before, X_df.shape[0], n_bad


def handle_negatives(X_df, mode):
    min_val = X_df.values.min()
    n_neg = int((X_df.values < 0).sum())
    shift_applied = 0.0
    if n_neg > 0:
        if mode == "error":
            raise ValueError(f"Found {n_neg} negative values and mode='error'.")
        elif mode == "clip":
            X_df = X_df.clip(lower=0.0)
        elif mode == "shift":
            shift = -min_val
            X_df = X_df + shift
            shift_applied = shift
    return X_df, n_neg, shift_applied


def apply_scaling(X_df, scale_mode):
    X_out = X_df.copy()
    if scale_mode == "none":
        return X_out
    elif scale_mode == "minmax":
        scaler = MinMaxScaler()
        X_out = pd.DataFrame(scaler.fit_transform(X_df.values),
                             index=X_df.index, columns=X_df.columns)
        return X_out
    elif scale_mode == "log1p":
        # ensure non-negativity first (log1p undefined for negative)
        X_out = np.log1p(X_df.clip(lower=0.0))
        X_out = pd.DataFrame(X_out, index=X_df.index, columns=X_df.columns)
        return X_out
    else:
        return X_out


def nmf_decompose(X_array, n_components, init_method, random_state, max_iter, tol=1e-4, solver='cd'):
    # Return model and W,H and errors. Using sklearn NMF interface.
    model = NMF(n_components=n_components, init=init_method, random_state=random_state,
                max_iter=max_iter, tol=tol, solver=solver)
    W = model.fit_transform(X_array)
    H = model.components_
    recon = W.dot(H)
    recon_err = np.linalg.norm(X_array - recon, ord='fro')
    model_err = getattr(model, "reconstruction_err_", None)
    return model, W, H, recon_err, model_err


def crossval_xgboost_eval(X_feat, y, xgb_params, n_splits, n_repeats, base_random_state):
    """
    Runs repeated stratified k-fold CV for XGBoost and returns:
      - df_metrics (per fold metrics)
      - summary (agg)
      - roc_fprs, roc_tprs (lists of arrays, per fold)
      - pr_recs, pr_precs (lists of arrays, per fold)
      - y_trues_all, y_probs_all, y_preds_all (concatenated lists)
    """
    metrics_list = []
    roc_fprs = []
    roc_tprs = []
    pr_recs = []
    pr_precs = []
    y_trues_all = []
    y_probs_all = []
    y_preds_all = []

    for r in range(n_repeats):
        seed = int(base_random_state + r)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_idx = 0
        for train_idx, test_idx in skf.split(X_feat, y):
            X_train, X_test = X_feat[train_idx], X_feat[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **xgb_params)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
            y_pred = (y_prob >= 0.5).astype(int)

            # store true/probs/preds for aggregated confusion matrix and PR curve
            y_trues_all.append(y_test)
            y_probs_all.append(y_prob)
            y_preds_all.append(y_pred)

            # metrics
            try:
                roc = roc_auc_score(y_test, y_prob)
            except Exception:
                roc = np.nan
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)

            # PR curve per fold
            try:
                precision, recall, _ = precision_recall_curve(y_test, y_prob)
                pr_precs.append(precision)
                pr_recs.append(recall)
            except Exception:
                pass

            metrics_list.append({
                "repeat": r, "fold": fold_idx, "roc_auc": roc, "accuracy": acc,
                "f1": f1, "precision": prec, "recall": rec
            })

            # store ROC curve
            try:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                roc_fprs.append(fpr)
                roc_tprs.append(tpr)
            except Exception:
                pass

            fold_idx += 1

    df_metrics = pd.DataFrame(metrics_list)
    summary = df_metrics.agg(['mean', 'std']).to_dict()

    return df_metrics, summary, roc_fprs, roc_tprs, pr_recs, pr_precs, y_trues_all, y_probs_all, y_preds_all


# ---------- Main experiment runner ----------
def run_experiment(exp_id, params, df, sample_col, response_col, feature_cols, outdir, global_opts):
    """
    params: dict with keys:
      - k (n_components)
      - neg_mode
      - w_mode ('raw' or 'relative')
      - scale (none/minmax/log1p)
      - init
    """
    start = time.time()
    res = {"exp_id": int(exp_id), **params}
    exp_dir = Path(outdir) / f"exp_{int(exp_id):04d}_k{params['k']}_neg{params['neg_mode']}_w{params['w_mode']}_scale{params['scale']}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data copies
    X_df = df[feature_cols].copy()
    sample_accession = df[sample_col].astype(str).copy()
    response_series = df[response_col].copy()

    # Convert numeric
    X_df = X_df.apply(pd.to_numeric, errors='coerce')

    # Remove NaN/inf rows
    X_df, sample_accession, response_series, n_before, n_after, n_bad = remove_nan_inf_rows(X_df, sample_accession, response_series)
    res.update({"n_rows_before": int(n_before), "n_rows_after": int(n_after), "n_bad_naninf_rows": int(n_bad)})

    if n_after == 0:
        res["status"] = "no_rows_left_after_naninf"
        return res

    # Apply scaling
    X_df = apply_scaling(X_df, params['scale'])

    # Handle negatives
    try:
        X_df, n_neg_values, shift_applied = handle_negatives(X_df, params['neg_mode'])
    except Exception as e:
        res["status"] = "negative_error"
        res["error"] = str(e)
        return res
    res.update({"n_negative_values_before": int(n_neg_values), "shift_applied": float(shift_applied)})

    # Prepare numpy array
    X_input = X_df.values
    n_samples, n_features = X_input.shape
    res.update({"n_samples": int(n_samples), "n_features": int(n_features)})

    # Adjust k if too large
    k = int(params['k'])
    if k >= min(n_samples, n_features):
        k = max(1, min(n_samples, n_features) - 1)
        res["k_adjusted"] = int(k)

    # NMF (with robust try)
    try:
        nmf_model, W, H, recon_err, model_err = nmf_decompose(
            X_input, k, params['init'], int(global_opts['random_state']),
            int(global_opts['max_iter']), tol=float(global_opts.get('tol', 1e-4)), solver=global_opts.get('solver', 'cd')
        )
    except Exception as e:
        res["status"] = "nmf_error"
        res["error"] = str(e)
        return res

    res.update({"recon_err": float(recon_err), "model_recon_err": float(model_err) if model_err is not None else None})

    # Build features for classifier
    df_W = pd.DataFrame(W, index=sample_accession.values, columns=[f"Comp_{i+1}" for i in range(W.shape[1])])
    if params['w_mode'] == 'raw':
        X_feats = df_W.values
    elif params['w_mode'] == 'relative':
        sums = df_W.sum(axis=1).replace(0, np.nan)
        df_W_rel = df_W.div(sums, axis=0).fillna(0.0)
        X_feats = df_W_rel.values
    else:
        X_feats = df_W.values

    # Save W, H
    df_W.to_csv(exp_dir / "W_samples_components.csv", index=True)
    pd.DataFrame(H, index=[f"Comp_{i+1}" for i in range(H.shape[0])], columns=feature_cols).to_csv(exp_dir / "H_components_features.csv", index=True)
    pd.DataFrame({"sample": sample_accession.values, response_col: response_series.values}).to_csv(exp_dir / "meta_samples.csv", index=False)

    # XGBoost params
    xgb_params = global_opts['xgb_params']

    # Run repeated CV and get metrics (extended outputs)
    y = response_series.values
    try:
        (df_metrics, metrics_summary,
         roc_fprs, roc_tprs,
         pr_recs, pr_precs,
         y_trues_all, y_probs_all, y_preds_all) = crossval_xgboost_eval(
            X_feats, y, xgb_params,
            int(global_opts['n_splits']), int(global_opts['n_repeats']), int(global_opts['random_state'])
        )
    except Exception as e:
        res["status"] = "xgb_error"
        res["error"] = str(e)
        return res

    # Save per-fold metrics
    df_metrics.to_csv(exp_dir / "xgb_cv_metrics_per_fold.csv", index=False)

    # Summarize metrics (mean and std)
    summary_stats = {}
    for metric in ["roc_auc", "accuracy", "f1", "precision", "recall"]:
        try:
            summary_stats[f"{metric}_mean"] = float(df_metrics[metric].mean())
            summary_stats[f"{metric}_std"] = float(df_metrics[metric].std())
        except Exception:
            summary_stats[f"{metric}_mean"] = None
            summary_stats[f"{metric}_std"] = None
    res.update(summary_stats)

    # ----- PLOT 1: ROC curves (fold-level + mean + std band) -----
    try:
        mean_fpr = np.linspace(0, 1, 200)
        interp_tprs = []
        for fpr, tpr in zip(roc_fprs, roc_tprs):
            try:
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                interp_tprs.append(tpr_interp)
            except Exception:
                pass

        if len(interp_tprs) > 0:
            mean_tpr = np.mean(interp_tprs, axis=0)
            std_tpr = np.std(interp_tprs, axis=0)

            plt.figure(figsize=(6, 6))
            for fpr, tpr in zip(roc_fprs, roc_tprs):
                try:
                    plt.plot(fpr, tpr, color="0.8", linewidth=0.8)
                except Exception:
                    pass
            plt.plot(mean_fpr, mean_tpr, color="C0", lw=2, label=f"mean ROC (AUC~{summary_stats.get('roc_auc_mean', np.nan):.3f})")
            plt.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0), np.minimum(mean_tpr + std_tpr, 1),
                             color="C0", alpha=0.2, label="±1 std")
            plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC (folds + mean) - exp {exp_id}")
            plt.legend(loc="lower right")
            plt.tight_layout()
            roc_plot_path = exp_dir / "roc_folds_mean.png"
            plt.savefig(roc_plot_path, dpi=150)
            plt.close()
            res["roc_plot"] = str(roc_plot_path)
    except Exception as e:
        print("Warning: error plotting ROC:", e)

    # ----- PLOT 2: Precision-Recall (fold-level + mean AP) -----
    try:
        mean_recall = np.linspace(0, 1, 200)
        interp_precisions = []
        for prec, rec in zip(pr_precs, pr_recs):
            try:
                rec_rev = rec[::-1]
                prec_rev = prec[::-1]
                prec_interp = np.interp(mean_recall, rec_rev, prec_rev, left=prec_rev[0], right=prec_rev[-1])
                interp_precisions.append(prec_interp)
            except Exception:
                pass
        if len(interp_precisions) > 0:
            mean_prec = np.mean(interp_precisions, axis=0)
            std_prec = np.std(interp_precisions, axis=0)
            try:
                y_all = np.concatenate(y_trues_all)
                p_all = np.concatenate(y_probs_all)
                ap_overall = average_precision_score(y_all, p_all)
            except Exception:
                ap_overall = np.nan

            plt.figure(figsize=(6, 6))
            for prec, rec in zip(pr_precs, pr_recs):
                try:
                    plt.plot(rec, prec, color="0.8", linewidth=0.8)
                except Exception:
                    pass
            plt.plot(mean_recall, mean_prec, color="C1", lw=2, label=f"mean PR (AP~{ap_overall:.3f})")
            plt.fill_between(mean_recall, np.maximum(mean_prec - std_prec, 0), np.minimum(mean_prec + std_prec, 1),
                             color="C1", alpha=0.2, label="±1 std")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"Precision-Recall (folds + mean) - exp {exp_id}")
            plt.legend(loc="lower left")
            plt.tight_layout()
            pr_plot_path = exp_dir / "pr_folds_mean.png"
            plt.savefig(pr_plot_path, dpi=150)
            plt.close()
            res["pr_plot"] = str(pr_plot_path)
    except Exception as e:
        print("Warning: error plotting PR:", e)

    # ----- PLOT 3: Aggregated confusion matrix (all CV folds concatenated) -----
    try:
        y_all_true = np.concatenate(y_trues_all)
        y_all_pred = np.concatenate(y_preds_all)
        cm = confusion_matrix(y_all_true, y_all_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        plt.figure(figsize=(4.5, 4))
        plt.imshow(cm_norm, interpolation='nearest', aspect='auto')
        plt.colorbar()
        plt.xticks([0, 1], ['pred_0', 'pred_1'])
        plt.yticks([0, 1], ['true_0', 'true_1'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, f"{cm[i, j]:d}\n({cm_norm[i, j]:.2f})", ha='center', va='center',
                         color='white' if cm_norm[i, j] > 0.5 else 'black', fontsize=9)
        plt.title(f"Confusion matrix (normalized) - exp {exp_id}")
        plt.tight_layout()
        cm_path = exp_dir / "confusion_matrix_cv_agg.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
        res["confusion_matrix_plot"] = str(cm_path)
    except Exception as e:
        print("Warning: error plotting confusion matrix:", e)

    # ----- PLOT 4: Feature importance from model trained on FULL W (train once on all data) -----
    if global_opts.get('save_models', False):
        try:
            full_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss", **xgb_params)
            full_model.fit(X_feats, y)
            joblib_dump(full_model, exp_dir / "xgb_full_model.joblib")
            importances = full_model.feature_importances_
            fi_indices = np.argsort(importances)[::-1]
            top_n = min(30, len(importances))
            top_indices = fi_indices[:top_n]
            top_feats = [df_W.columns[i] for i in top_indices]
            top_vals = importances[top_indices]
            plt.figure(figsize=(6, max(3, top_n*0.15)))
            plt.barh(range(len(top_feats))[::-1], top_vals[::-1])
            plt.yticks(range(len(top_feats))[::-1], top_feats[::-1])
            plt.xlabel("Feature importance (XGBoost)")
            plt.title(f"Top feature importances - exp {exp_id}")
            plt.tight_layout()
            fi_path = exp_dir / "feature_importances_top.png"
            plt.savefig(fi_path, dpi=150)
            plt.close()
            res["feature_importance_plot"] = str(fi_path)
        except Exception as e:
            print("Warning: error training full XGB model for feature importances:", e)

    # Save roc_mean.csv for global comparison (if available)
    if len(roc_fprs) > 0:
        mean_fpr = np.linspace(0, 1, 200)
        interp_tprs = []
        for fpr, tpr in zip(roc_fprs, roc_tprs):
            try:
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                interp_tprs.append(tpr_interp)
            except Exception:
                pass
        if len(interp_tprs) > 0:
            mean_tpr = np.mean(interp_tprs, axis=0)
            std_tpr = np.std(interp_tprs, axis=0)
            roc_df = pd.DataFrame({"fpr": mean_fpr, "tpr_mean": mean_tpr, "tpr_std": std_tpr})
            roc_df.to_csv(exp_dir / "roc_mean.csv", index=False)

    # Save experiment summary json
    res["status"] = "ok"
    res["time_elapsed_s"] = time.time() - start
    summary_path = exp_dir / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(res, f, indent=2, default=float)

    return res


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Read CSV and identify columns
    df = safe_read_csv(args.file)
    col_names = df.columns.tolist()
    if len(col_names) < 3:
        raise ValueError("Input CSV must have at least 3 columns: Sample_geo_accession, features..., Response")

    sample_col = col_names[0]
    response_col = col_names[-1]
    feature_cols = col_names[1:-1]

    # xgb params
    if args.xgb_params:
        xgb_params = json.loads(args.xgb_params)
    else:
        xgb_params = {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1, "verbosity": 0}

    # Build experiment grid
    grid = list(itertools.product(args.ks, args.neg_modes, args.w_modes, args.scales))
    total_exps = len(grid)
    if total_exps == 0:
        raise ValueError("Grid of experiments is empty. Check arguments.")
    if total_exps > args.max_experiments:
        raise SystemExit(f"Grid too large ({total_exps} experiments). Abort. Set --max_experiments higher if intended.")
    experiments = []
    exp_id = 0
    for (k, neg_mode, w_mode, scale) in grid:
        params = {"k": int(k), "neg_mode": neg_mode, "w_mode": w_mode, "scale": scale, "init": "nndsvda"}
        experiments.append((exp_id, params))
        exp_id += 1

    print(f"Total experiments to run: {len(experiments)}")
    print("First experiments sample:")
    for e in experiments[:10]:
        print(e)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = outdir / f"run_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)

    global_opts = {
        "xgb_params": xgb_params,
        "n_splits": args.n_splits,
        "n_repeats": args.n_repeats,
        "random_state": args.random_state,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "solver": args.solver,
        "save_models": args.save_models
    }

    run_func = partial(run_experiment, df=df, sample_col=sample_col, response_col=response_col,
                       feature_cols=feature_cols, outdir=results_dir, global_opts=global_opts)

    results = []
    if args.n_jobs == 1:
        for exp_id, params in tqdm(experiments, desc="Running experiments"):
            r = run_func(exp_id=exp_id, params=params)
            results.append(r)
    else:
        results = Parallel(n_jobs=args.n_jobs)(
            delayed(run_func)(exp_id=exp_id, params=params) for exp_id, params in experiments
        )

    # Save global results summary CSV
    df_results = pd.DataFrame(results)
    df_results_path = results_dir / "experiments_summary_table.csv"
    df_results.to_csv(df_results_path, index=False)
    print(f"Saved experiments summary to: {df_results_path}")

    # Filter OK experiments
    df_ok = df_results[df_results['status'] == 'ok'].copy()
    if df_ok.shape[0] == 0:
        print("No successful experiments to plot.")
        return

    # Plot: bar chart of mean ROC AUC with error bars
    metrics = ["roc_auc_mean", "accuracy_mean", "f1_mean", "precision_mean", "recall_mean"]
    for m in metrics:
        if m not in df_ok.columns:
            df_ok[m] = np.nan

    plt.figure(figsize=(max(8, len(df_ok)*0.6), 6))
    x = np.arange(len(df_ok))
    y = df_ok['roc_auc_mean'].astype(float).values
    yerr = df_ok['roc_auc_std'].astype(float).values if 'roc_auc_std' in df_ok.columns else np.zeros_like(y)
    labels = df_ok.apply(lambda r: f"k={int(r['k'])}\n{r['neg_mode']}/{r['w_mode']}/{r['scale']}", axis=1)
    plt.bar(x, y, yerr=yerr)
    plt.xticks(x, labels, rotation=45, ha='right')
    plt.ylabel("ROC AUC (mean)")
    plt.ylim(0, 1.0)
    plt.title("Comparación de experimentos - ROC AUC (mean ± std)")
    plt.tight_layout()
    plt.savefig(results_dir / "compare_roc_auc_mean.png", dpi=150)
    plt.close()
    print("Saved: compare_roc_auc_mean.png")

    # Save metric means table
    metric_means_df = df_ok[['exp_id', 'k', 'neg_mode', 'w_mode', 'scale'] + metrics].copy()
    metric_means_df.to_csv(results_dir / "metric_means_by_experiment.csv", index=False)

    # Plot grouped metrics (first N experiments)
    N = min(12, metric_means_df.shape[0])
    subset = metric_means_df.head(N)
    labels = subset.apply(lambda r: f"k={int(r['k'])}\n{r['neg_mode']}/{r['w_mode']}/{r['scale']}", axis=1).tolist()
    ind = np.arange(N)
    width = 0.15
    plt.figure(figsize=(max(10, N*1.2), 6))
    for i, metric in enumerate(metrics):
        vals = subset[metric].astype(float).values
        plt.bar(ind + i*width, vals, width, label=metric)
    plt.xticks(ind + width*2, labels, rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.title("Comparación múltiple de métricas (primeros experiments)")
    plt.tight_layout()
    plt.savefig(results_dir / "compare_metrics_grouped_firstN.png", dpi=150)
    plt.close()
    print("Saved: compare_metrics_grouped_firstN.png")

    # Top experiments ROC comparison (top 3 by ROC AUC mean)
    try:
        df_ok_sorted = df_ok.sort_values(by='roc_auc_mean', ascending=False)
        topk = min(3, df_ok_sorted.shape[0])
        top_exps = df_ok_sorted.head(topk)
        plt.figure(figsize=(6, 6))
        any_plotted = False
        for idx, row in top_exps.iterrows():
            exp_id = int(row['exp_id'])
            folder_pattern = f"exp_{exp_id:04d}_k{int(row['k'])}_neg{row['neg_mode']}_w{row['w_mode']}_scale{row['scale']}"
            roc_csv = results_dir / folder_pattern / "roc_mean.csv"
            if roc_csv.exists():
                roc_df = pd.read_csv(roc_csv)
                plt.plot(roc_df['fpr'], roc_df['tpr_mean'], lw=2, label=f"exp{exp_id} k={int(row['k'])} AUC={row['roc_auc_mean']:.3f}")
                any_plotted = True
        if any_plotted:
            plt.plot([0,1],[0,1], linestyle='--', color='gray')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.title(f"Top {topk} experiments ROC mean")
            plt.legend(loc='lower right')
            plt.tight_layout()
            plt.savefig(results_dir / "top_experiments_roc_comparison.png", dpi=150)
            plt.close()
            print("Saved: top_experiments_roc_comparison.png")
        else:
            print("No roc_mean.csv found for top experiments; skipping ROC comparison plot.")
    except Exception as e:
        print("Warning: couldn't create top experiments ROC comparison:", e)

    print("All done. Results in:", results_dir)


if __name__ == "__main__":
    main()
