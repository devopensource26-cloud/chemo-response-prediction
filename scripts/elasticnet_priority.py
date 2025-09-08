# -*- coding: utf-8 -*-
"""
Priority Stability Selection con Elastic Net + SAGA
- Menor penalización a genes con mayor priority_log
- Emula 'penalty.factor' de glmnet escalando columnas tras StandardScaler

Entradas:
  - data_path: CSV con expresión + ['Sample_geo_accession', 'Response', genes...]
  - priority_path: CSV con ['Gene','PubMed_hits','priority_log','priority_binary']

Salida:
  - stability_frequencies_priority.csv
  - selected_features_priority.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit

# ========================== Configuración ==========================
data_path = Path("./data/processed/data_GSE205568_mapped_clean.csv")
priority_path = Path("./data/processed/genes_with_priority_variants.csv")

N_RESAMPLES   = 60
SAMPLE_FRAC   = 0.7
FEATURE_FRAC  = 0.6
L1_RATIO      = 0.5
C_ENET        = 0.2
THRESHOLD     = 0.70
RANDOM_STATE  = 42
N_JOBS        = -1
MAX_ITER      = 1500
TOL           = 1e-3

# Escalamiento por prioridad
PRIORITY_GAMMA = 0.5
FACTOR_MIN     = 1.0
FACTOR_MAX     = 3.0


# ======================== Utilidades ==========================
def make_priority_factors(genes_in_X: pd.Index, df_pri: pd.DataFrame) -> pd.Series:
    dfp = df_pri.copy()
    dfp.columns = dfp.columns.str.strip()
    if "Gene" not in dfp.columns or "priority_log" not in dfp.columns:
        raise ValueError("priority_path debe tener columnas 'Gene' y 'priority_log'.")

    dfp = dfp[["Gene", "priority_log"]].dropna()
    dfp["Gene"] = dfp["Gene"].astype(str)

    # z-score en los genes con prioridad
    z = (dfp["priority_log"] - dfp["priority_log"].mean()) / (dfp["priority_log"].std(ddof=0) + 1e-12)
    factors = np.exp(PRIORITY_GAMMA * z)
    factors = np.clip(factors, FACTOR_MIN, FACTOR_MAX)
    dfp["priority_factor"] = factors

    pri_map = dfp.set_index("Gene")["priority_factor"]
    out = pd.Series(1.0, index=genes_in_X, dtype=float)
    intersect = genes_in_X.intersection(pri_map.index)
    out.loc[intersect] = pri_map.loc[intersect].astype(float)
    return out


def fast_priority_stability_enet(
    X: pd.DataFrame,
    y: pd.Series,
    column_scale: pd.Series,
    n_resamples: int = N_RESAMPLES,
    sample_frac: float = SAMPLE_FRAC,
    feature_frac: float = FEATURE_FRAC,
    l1_ratio: float = L1_RATIO,
    C: float = C_ENET,
    max_iter: int = MAX_ITER,
    tol: float = TOL,
    threshold: float = THRESHOLD,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    verbose: int = 1,
):
    # Escalado global
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    # Aplicar factores de prioridad
    col_order = pd.Index(X.columns)
    scale_vec = column_scale.reindex(col_order).fillna(1.0).to_numpy(dtype=np.float32)
    Xs = Xs * scale_vec

    n, p = Xs.shape
    features = np.array(X.columns)
    rng = np.random.default_rng(random_state)

    # Submuestreo estratificado de filas
    sss = StratifiedShuffleSplit(n_splits=n_resamples, train_size=sample_frac, random_state=random_state)
    row_splits = list(sss.split(np.zeros(n), y.values))

    # Submuestreo de columnas
    m = max(1, int(np.ceil(feature_frac * p)))
    col_subsets = [
        np.sort(rng.choice(p, size=m, replace=False)) if m < p else np.arange(p)
        for _ in range(n_resamples)
    ]

    def _fit_one(k: int):
        idx_train, _ = row_splits[k]
        cols_k = col_subsets[k]
        Xk = Xs[idx_train][:, cols_k]
        yk = y.values[idx_train]
        clf = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=l1_ratio,
            C=C,
            max_iter=max_iter,
            tol=tol,
            fit_intercept=True,
            class_weight="balanced",
        )
        clf.fit(Xk, yk)
        sel_mask = np.abs(clf.coef_.ravel()) > 1e-8
        return cols_k[sel_mask]

    if verbose:
        print(f"[Priority Stability] n={n}, p={p} | resamples={n_resamples}, "
              f"sample_frac={sample_frac}, feature_frac={feature_frac}, "
              f"l1_ratio={l1_ratio}, C={C}, threshold={threshold}")

    selected_lists = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_one)(k) for k in range(n_resamples)
    )

    counts = np.zeros(p, dtype=np.int32)
    for sel_idx in selected_lists:
        counts[sel_idx] += 1
    freq = counts / n_resamples
    freq_s = pd.Series(freq, index=features).sort_values(ascending=False)

    selected_features = freq_s.index[freq_s >= threshold].tolist()
    if verbose:
        print(f"[Priority Stability] Seleccionadas (>= {threshold:.2f}): {len(selected_features)} / {p}")
    return selected_features, freq_s


# ============================ Main =============================
if __name__ == "__main__":
    # Data principal
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()
    y = df["Response"].astype(int)
    X = df.drop(columns=[c for c in ["Response", "Sample_geo_accession"] if c in df.columns])

    print(f"[INFO] Data: {X.shape}, clases: {y.value_counts().to_dict()}")

    # Data prioridad
    df_pri = pd.read_csv(priority_path)
    pri_factors = make_priority_factors(pd.Index(X.columns.astype(str)), df_pri)

    # Ejecutar Priority Stability
    selected, freq = fast_priority_stability_enet(
        X, y,
        column_scale=pri_factors,
        n_resamples=N_RESAMPLES,
        sample_frac=SAMPLE_FRAC,
        feature_frac=FEATURE_FRAC,
        l1_ratio=L1_RATIO,
        C=C_ENET,
        threshold=THRESHOLD,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=1
    )

    # Guardar resultados
    freq.to_csv("./data/processed/stability_frequencies_priority7.csv")
    pd.Series(selected, name="selected_features7").to_csv("selected_features_priority7.csv", index=False)

    print("\n✅ Archivos generados:")
    print(" - stability_frequencies_priority7.csv")
    print(" - selected_features_priority7.csv")
