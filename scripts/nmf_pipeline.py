#!/usr/bin/env python3
# nmf_pipeline.py
# Pipeline NMF completo:
# - elimina filas con NaN / inf
# - maneja negativos (shift | clip | error)
# - ajusta NMF y guarda resultados
#
# Requisitos:
# pip install pandas numpy scikit-learn matplotlib

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Pipeline NMF para data_GSE205568_quantile_corrected_Biopsy.csv")
    p.add_argument("--file", "-f", type=str, required=True, help="Ruta al CSV (ej. data_GSE205568_quantile_corrected_Biopsy.csv)")
    p.add_argument("--n_components", "-k", type=int, default=10, help="Número de componentes K inicial")
    p.add_argument("--neg_mode", "-m", type=str, choices=["shift", "clip", "error"], default="shift",
                   help="Cómo manejar valores negativos antes de NMF: 'shift' suma |min|, 'clip' pone negativos a 0, 'error' lanza excepción")
    p.add_argument("--max_iter", type=int, default=1000, help="Max iter para NMF")
    p.add_argument("--init", type=str, default="nndsvda", help="Método init para NMF (sklearn: 'random', 'nndsvd', 'nndsvda', ...)")
    p.add_argument("--random_state", type=int, default=42, help="Random state para reproducibilidad")
    p.add_argument("--results_dir", "-o", type=str, default="results/nmf/", help="Directorio para guardar resultados")
    p.add_argument("--evaluate_k", nargs="*", type=int, default=None, help="(Opcional) lista de K para evaluar reconstrucción (ej. --evaluate_k 2 3 5 8 10)")
    p.add_argument("--top_k_features", type=int, default=10, help="Cuántas features top mostrar por componente")
    return p.parse_args()

def evaluate_k_values(X_array, k_list, init_method='nndsvda', random_state=42, max_iter=1000):
    results = {}
    print("\nEvaluando K candidatos para reconstrucción (Frobenius norm):")
    for k in k_list:
        if k < 1:
            continue
        if k >= min(X_array.shape):
            print(f"  K={k} inválido (>= min(n_samples, n_features)={min(X_array.shape)}). Omitiendo.")
            continue
        model = NMF(n_components=k, init=init_method, random_state=random_state, max_iter=max_iter)
        W = model.fit_transform(X_array)
        H = model.components_
        recon = W.dot(H)
        frob = np.linalg.norm(X_array - recon, ord='fro')
        results[k] = {'frob': frob, 'sklearn_reconstruction_err': model.reconstruction_err_}
        print(f"  K={k} -> frob={frob:.6f}, sklearn_reconstruction_err={model.reconstruction_err_:.6f}")
    return results

def main():
    args = parse_args()

    file_path = args.file
    n_components = args.n_components
    neg_mode = args.neg_mode
    max_iter = args.max_iter
    init_method = args.init
    random_state = args.random_state
    results_dir = args.results_dir
    evaluate_list = args.evaluate_k
    top_k = args.top_k_features

    os.makedirs(results_dir, exist_ok=True)

    # 1) Carga
    if not os.path.exists(file_path):
        print(f"ERROR: no se encontró el archivo '{file_path}'. Ajusta --file y vuelve a intentar.")
        sys.exit(1)

    print(f"Cargando archivo: {file_path}")
    df = pd.read_csv(file_path, index_col=False)
    print("Archivo cargado. Dimensiones:", df.shape)

    # 2) Identificar columnas según prompt
    col_names = df.columns.tolist()
    if len(col_names) < 3:
        raise ValueError("El archivo tiene menos de 3 columnas; revisa el formato.")

    sample_col = col_names[0]
    response_col = col_names[-1]
    feature_cols = col_names[1:-1]  # desde la segunda hasta la penúltima

    print(f"Sample column: {sample_col}")
    print(f"Response column: {response_col}")
    print(f"Nº features: {len(feature_cols)}")

    sample_accession = df[sample_col].astype(str).copy()
    response_series = df[response_col].copy()
    X = df[feature_cols].copy()

    # 3) Convertir a numérico (forzar NaN si no convertible)
    X = X.apply(pd.to_numeric, errors='coerce')

    # 4) Eliminar filas con NaN o inf (por fila)
    finite_mask = np.isfinite(X.values).all(axis=1)  # True si toda la fila es finita
    n_total_rows_before = X.shape[0]
    n_bad_rows = int((~finite_mask).sum())

    if n_bad_rows > 0:
        print(f"Se detectaron {n_bad_rows} filas con NaN o inf. Estas filas se eliminarán del dataset.")
        # Filtrar X y también las columnas de metadatos correspondientes
        X = X.loc[finite_mask].reset_index(drop=True)
        sample_accession = sample_accession.loc[finite_mask].reset_index(drop=True)
        response_series = response_series.loc[finite_mask].reset_index(drop=True)
    else:
        print("No se detectaron NaN ni inf en las filas de features.")

    print(f"Filas antes: {n_total_rows_before}, filas después: {X.shape[0]}")

    # 5) Revisar negativos
    min_val = X.values.min()
    n_neg_values = int((X.values < 0).sum())
    print(f"\nValor mínimo en X antes de corrección: {min_val}")
    print(f"Cantidad total de valores negativos en X: {n_neg_values}")

    if n_neg_values > 0:
        if neg_mode == "error":
            raise ValueError(f"Se encontraron {n_neg_values} valores negativos en X. Modo 'error' seleccionado -> abortando.")
        elif neg_mode == "clip":
            print("Modo 'clip' seleccionado: los valores negativos serán puestos a 0 (np.clip).")
            X = X.clip(lower=0.0)
            new_min = X.values.min()
            print(f"Valor mínimo en X después de clip: {new_min}")
        elif neg_mode == "shift":
            shift = -min_val
            print(f"Modo 'shift' seleccionado: se aplicará un shift de {shift} para forzar no-negatividad.")
            X = X + shift
            new_min = X.values.min()
            print(f"Valor mínimo en X después del shift: {new_min}")
    else:
        print("No hay valores negativos. No se aplica corrección.")

    # 6) Convertir a numpy para NMF
    X_input = X.values
    n_samples, n_features = X_input.shape
    print(f"\nMatriz X lista para NMF: samples={n_samples}, features={n_features}")

    # 7) Ajustar n_components si fuera inválido
    if n_components >= min(n_samples, n_features):
        max_allowed = min(n_samples, n_features) - 1
        print(f"Advertencia: n_components={n_components} debe ser < min(n_samples, n_features)={min(n_samples, n_features)}.")
        n_components = max(1, max_allowed)
        print(f"Ajustado n_components a {n_components}.")

    # 8) Evaluación opcional de varios K
    if evaluate_list:
        evaluate_k_values(X_input, evaluate_list, init_method=init_method, random_state=random_state, max_iter=max_iter)

    # 9) Instanciar y ajustar NMF con manejo de excepciones
    nmf = NMF(n_components=n_components, init=init_method, random_state=random_state, max_iter=max_iter)
    try:
        W_samples_components = nmf.fit_transform(X_input)  # (n_samples, n_components)
        H_components_features = nmf.components_             # (n_components, n_features)
    except ValueError as e:
        msg = str(e)
        print("Error durante NMF:", msg)
        if ('Negative values' in msg) or ('non-negative' in msg and 'Negative' in msg):
            # Intento de recuperación: clip y reintentar
            print("Intentando corrección automática con np.clip y reintento...")
            X_input = np.clip(X_input, 0.0, None)
            W_samples_components = nmf.fit_transform(X_input)
            H_components_features = nmf.components_
            print("Reintento con clip exitoso.")
        else:
            raise

    # 10) Resultados y métricas
    recon = W_samples_components.dot(H_components_features)
    recon_error = np.linalg.norm(X_input - recon, ord='fro')
    print("\nNMF completado.")
    print(f"n_components = {n_components}")
    print(f"Shape W (samples x components): {W_samples_components.shape}")
    print(f"Shape H (components x features): {H_components_features.shape}")
    print(f"Frobenius reconstruction error ||X - WH||_F = {recon_error:.6f}")
    print(f"Sklearn model.reconstruction_err_ = {nmf.reconstruction_err_:.6f}")

    normX = np.linalg.norm(X_input, ord='fro')
    explained_frac = 1.0 - (recon_error / normX)
    print(f"||X||_F = {normX:.6f}")
    print(f"Fracción explicada por WH: {explained_frac:.2%}")

    # 11) Convertir a DataFrames legibles
    component_names = [f'Component_{i+1}' for i in range(n_components)]
    df_W = pd.DataFrame(W_samples_components, index=sample_accession.values, columns=component_names)
    df_H = pd.DataFrame(H_components_features, index=component_names, columns=feature_cols)

    # Mostrar primeras filas
    print("\nPrimeras 5 filas de W (contribución de componentes en muestras):")
    print(df_W.head())

    print("\nPrimeras 5 filas de H (peso de features en cada componente):")
    print(df_H.head())

    # 12) Asociar W con Response y calcular medias por grupo
    df_W_with_response = df_W.copy()
    df_W_with_response[response_col] = response_series.values
    try:
        group_means = df_W_with_response.groupby(response_col).mean()
        print(f"\nMedia de activaciones de componentes por valor de '{response_col}':")
        print(group_means)
    except Exception as e:
        print("No se pudo agrupar por response:", e)

    # 13) Guardar resultados
    df_W.to_csv(os.path.join(results_dir, 'nmf_W_samples_components.csv'), index=True)
    df_H.to_csv(os.path.join(results_dir, 'nmf_H_components_features.csv'), index=True)
    df_W_with_response.to_csv(os.path.join(results_dir, 'nmf_W_with_response.csv'), index=True)
    print(f"\nResultados guardados en: {results_dir}")

    # 14) Heatmap de H (guardar)
    try:
        plt.figure(figsize=(10, max(4, n_components*0.5)))
        plt.imshow(df_H.values, aspect='auto')
        plt.colorbar()
        plt.yticks(ticks=np.arange(n_components), labels=component_names)
        plt.xlabel('Features (orden original)')
        plt.title('Matriz H (components x features) - heatmap')
        plt.tight_layout()
        heatmap_path = os.path.join(results_dir, 'nmf_H_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"Heatmap de H guardado en: {heatmap_path}")
    except Exception as e:
        print("No se pudo generar el heatmap de H:", e)

    # 15) Top features por componente
    top_features = {}
    print(f"\nTop {top_k} features por componente:")
    for comp in component_names:
        sorted_feats = df_H.loc[comp].sort_values(ascending=False)
        top = sorted_feats.head(top_k).index.tolist()
        top_features[comp] = top
        print(f"  {comp}: {top}")

    # Guardar top features como CSV simple
    df_top = pd.DataFrame.from_dict(top_features, orient='index')
    df_top.to_csv(os.path.join(results_dir, 'nmf_top_features_per_component.csv'), index=True, header=False)
    print(f"\nTop features por componente guardadas en: {os.path.join(results_dir, 'nmf_top_features_per_component.csv')}")

    print("\nPipeline finalizado correctamente.")

if __name__ == "__main__":
    main()

# #!/usr/bin/env python3
# """
# nmf_pipeline.py
# Implementación de NMF para el dataset data_GSE205568_quantile_corrected_Biopsy.csv

# Requisitos:
# pip install pandas numpy scikit-learn matplotlib
# """

# import os
# import sys
# import numpy as np
# import pandas as pd
# from sklearn.decomposition import NMF
# from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt

# # ---------------------------
# # Configuración / Parámetros
# # ---------------------------
# file_path = './data/raw/data_GSE205568_quantile_corrected_Biopsy.csv'  # cambiar si hace falta
# n_components = 10           # K inicial (ajustable)
# random_state = 42
# max_iter = 1000
# init_method = 'nndsvda'     # buena práctica para convergencia
# save_results = True         # guarda CSVs de salida
# results_dir = './results/nmf' # carpeta para guardar resultados
# # ---------------------------

# # 1) Carga de datos
# if not os.path.exists(file_path):
#     print(f"ERROR: no se encontró el archivo '{file_path}'. Ajusta file_path y vuelve a intentar.")
#     sys.exit(1)

# df = pd.read_csv(file_path, index_col=False)
# print("Archivo cargado. Dimensiones:", df.shape)

# # 2) Identificar columnas tal como pide el prompt
# # - Primera columna: Sample_geo_accession (identificador)
# # - Última columna: Response
# # - Features: desde la segunda hasta la penúltima (automáticamente)
# col_names = df.columns.tolist()
# if len(col_names) < 3:
#     raise ValueError("El archivo tiene menos de 3 columnas; revisa el formato.")

# sample_col = col_names[0]
# response_col = col_names[-1]
# feature_cols = col_names[1:-1]  # desde segunda hasta penúltima

# print(f"Sample column: {sample_col}")
# print(f"Response column: {response_col}")
# print(f"Nº features: {len(feature_cols)}")

# # Guardar columnas meta
# sample_accession = df[sample_col].astype(str).copy()
# response_series = df[response_col].copy()

# # Construir matriz X
# X = df[feature_cols].copy()

# # 3) Limpieza y verificación
# # Convertir a numérico (por si hay strings), forzar NaNs donde no convertible
# X = X.apply(pd.to_numeric, errors='coerce')

# # Identificar filas con NaN o inf (o -inf)
# # np.isfinite devuelve False para NaN, inf y -inf
# finite_mask = np.isfinite(X.values).all(axis=1)  # True si toda la fila es finita

# n_total_rows_before = X.shape[0]
# n_bad_rows = int((~finite_mask).sum())

# if n_bad_rows > 0:
#     print(f"Se detectaron {n_bad_rows} filas con NaN o inf. Estas filas se eliminarán del dataset.")
#     # Filtrar X y también las columnas de metadatos correspondientes
#     X = X.loc[finite_mask].reset_index(drop=True)
#     sample_accession = sample_accession.loc[finite_mask].reset_index(drop=True)
#     response_series = response_series.loc[finite_mask].reset_index(drop=True)
# else:
#     print("No se detectaron NaN ni inf en las filas de features.")

# print(f"Filas antes: {n_total_rows_before}, filas después: {X.shape[0]}")

# # (Opcional) Escalado: NMF no requiere estandarización, pero a veces MinMax puede ayudar.
# # Aquí lo dejamos como opción comentada.
# # scaler = MinMaxScaler()
# # X_scaled = pd.DataFrame(scaler.fit_transform(X), index=X.index, columns=X.columns)
# # X_input = X_scaled.values
# X_input = X.values

# n_samples, n_features = X_input.shape
# print(f"Matriz X: samples={n_samples}, features={n_features}")

# # Comprobación de n_components válido
# max_components_allowed = min(n_samples, n_features) - 1
# if n_components >= min(n_samples, n_features):
#     print(f"Advertencia: n_components={n_components} debe ser < min(n_samples, n_features)={min(n_samples, n_features)}. Ajustando a {max_components_allowed}.")
#     n_components = max(1, max_components_allowed)

# # 4) (Opcional) Búsqueda rápida sobre varios K usando error de reconstrucción
# def evaluate_k_values(X_array, k_list, random_state=42):
#     errs = {}
#     for k in k_list:
#         model = NMF(n_components=k, init=init_method, random_state=random_state, max_iter=max_iter)
#         W = model.fit_transform(X_array)
#         H = model.components_
#         recon = W.dot(H)
#         frob = np.linalg.norm(X_array - recon, ord='fro')
#         errs[k] = {'reconstruction_frobenius': frob, 'nmf_reconstruction_err_attr': model.reconstruction_err_}
#         print(f"K={k} -> frob={frob:.6f}, model.reconstruction_err_={model.reconstruction_err_:.6f}")
#     return errs

# # Si quieres evaluar varios K, descomenta la siguiente línea:
# # candidate_k = [2, 3, 5, 8, 10, 15]
# # eval_results = evaluate_k_values(X_input, candidate_k, random_state=random_state)

# # 5) Ajuste final con el n_components elegido
# nmf = NMF(n_components=n_components, init=init_method, random_state=random_state, max_iter=max_iter)
# W_samples_components = nmf.fit_transform(X_input)  # shape: (n_samples, n_components)
# H_components_features = nmf.components_               # shape: (n_components, n_features)
# recon = W_samples_components.dot(H_components_features)
# recon_error = np.linalg.norm(X_input - recon, ord='fro')

# print("\nNMF completado.")
# print(f"n_components = {n_components}")
# print(f"Shape W (samples x components): {W_samples_components.shape}")
# print(f"Shape H (components x features): {H_components_features.shape}")
# print(f"Frobenius reconstruction error ||X - WH||_F = {recon_error:.6f}")
# print(f"Sklearn model.reconstruction_err_ = {nmf.reconstruction_err_:.6f}")

# # 6) Convertir a DataFrames legibles
# component_names = [f'Component_{i+1}' for i in range(n_components)]
# df_W = pd.DataFrame(W_samples_components, index=sample_accession, columns=component_names)
# df_H = pd.DataFrame(H_components_features, index=component_names, columns=feature_cols)

# # Mostrar primeras filas
# print("\nPrimeras 5 filas de W (contribución de componentes en muestras):")
# print(df_W.head())

# print("\nPrimeras 5 filas de H (peso de features en cada componente):")
# print(df_H.head())

# # 7) Asociar W con Response para análisis rápido (media por grupo)
# df_W_with_response = df_W.copy()
# df_W_with_response[response_col] = response_series.values
# group_means = df_W_with_response.groupby(response_col).mean()
# print(f"\nMedia de activaciones de componentes por valor de '{response_col}':")
# print(group_means)

# # 8) Guardar resultados
# if save_results:
#     os.makedirs(results_dir, exist_ok=True)
#     df_W.to_csv(os.path.join(results_dir, 'nmf_W_samples_components.csv'), index=True)
#     df_H.to_csv(os.path.join(results_dir, 'nmf_H_components_features.csv'), index=True)
#     df_W_with_response.to_csv(os.path.join(results_dir, 'nmf_W_with_response.csv'), index=True)
#     print(f"\nResultados guardados en la carpeta: {results_dir}")

# # 9) Visualizaciones rápidas (opcionales)
# try:
#     # Heatmap simple de H (components x features) - si muchas features, puede ser denso
#     plt.figure(figsize=(10, 6))
#     plt.imshow(df_H.values, aspect='auto')
#     plt.colorbar()
#     plt.yticks(ticks=np.arange(n_components), labels=component_names)
#     plt.xlabel('Features (orden original)')
#     plt.title('Matriz H (componentes x features) - heatmap')
#     plt.tight_layout()
#     heatmap_path = os.path.join(results_dir, 'nmf_H_heatmap.png')
#     plt.savefig(heatmap_path)
#     plt.close()
#     print(f"Heatmap de H guardado en: {heatmap_path}")
# except Exception as e:
#     print("Ocurrió un error generando la visualización (posible por número de features):", e)

# # 10) (Opcional) Lista de features top por componente para interpretación
# top_k = 10
# top_features = {}
# for comp in component_names:
#     sorted_feats = df_H.loc[comp].sort_values(ascending=False)
#     top_features[comp] = sorted_feats.head(top_k).index.tolist()

# print("\nTop features por componente (top 10):")
# for comp in component_names:
#     print(f"{comp}: {top_features[comp]}")

# # Fin
# print("\nPipeline finalizado.")
