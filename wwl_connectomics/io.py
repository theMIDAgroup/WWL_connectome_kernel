"""Loading and normalizing FC/DTI matrices, anagrafica tables, and scenario file discovery."""

import glob
import os

import numpy as np
import pandas as pd


def load_matrix(source):
    if isinstance(source, (str, os.PathLike)):
        try:
            df = pd.read_csv(source, sep=";", header=None, index_col=None)
            return df.values.astype(np.float64)
        except Exception:
            df = pd.read_csv(source, sep=",", header=None, index_col=None)
            return df.values.astype(np.float64)
    elif isinstance(source, pd.DataFrame):
        return source.values.astype(np.float64)
    elif isinstance(source, np.ndarray):
        return source.astype(np.float64)
    else:
        raise TypeError(f"Tipo non supportato: {type(source)}")


def normalize_fc(FC_list, method="none"):
    FC_list = [load_matrix(fc) for fc in FC_list]
    for fc in FC_list:
        np.fill_diagonal(fc, 0)

    if method == "none":
        return FC_list

    if method == "zscore_global":
        stack = np.vstack(FC_list)
        mu    = stack.mean(axis=0)
        sigma = stack.std(axis=0) + 1e-8
        out = []
        for fc in FC_list:
            fc_norm = (fc - mu) / sigma
            np.fill_diagonal(fc_norm, 0)
            out.append(fc_norm)
        return out

    if method == "zscore_subject":
        out = []
        for fc in FC_list:
            mu    = fc.mean(axis=1, keepdims=True)
            sigma = fc.std(axis=1, keepdims=True) + 1e-8
            fc_norm = (fc - mu) / sigma
            np.fill_diagonal(fc_norm, 0)
            out.append(fc_norm)
        return out

    if method == "minmax_global":
        stack = np.vstack(FC_list)
        vmin, vmax = stack.min(), stack.max()
        out = []
        for fc in FC_list:
            fc_norm = (fc - vmin) / (vmax - vmin + 1e-8)
            np.fill_diagonal(fc_norm, 0)
            out.append(fc_norm)
        return out

    raise ValueError(f"method non valido: {method}")


def normalize_dti(DTI_list, method="minmax_subject"):
    DTI_list = [load_matrix(dti) for dti in DTI_list]
    for dti in DTI_list:
        np.fill_diagonal(dti, 0)
        dti[dti < 0] = 0

    if method == "none":
        return DTI_list

    if method == "minmax_subject":
        out = []
        for dti in DTI_list:
            vmax = dti.max()
            dti_norm = dti / vmax if vmax > 0 else dti.copy()
            out.append(dti_norm)
        return out

    if method == "minmax_global":
        stack = np.concatenate([d.ravel() for d in DTI_list])
        vmax  = stack.max()
        out   = [dti / (vmax + 1e-8) for dti in DTI_list]
        return out

    raise ValueError(f"method non valido: {method}")


def load_and_normalize(fc_sources, dti_sources, fc_method="none", dti_method="minmax_subject"):
    assert len(fc_sources) == len(dti_sources), \
        "FC e DTI devono avere lo stesso numero di soggetti"
    print(f"Caricamento {len(fc_sources)} soggetti...")
    FC_list  = normalize_fc(fc_sources, method=fc_method)
    DTI_list = normalize_dti(dti_sources, method=dti_method)
    return FC_list, DTI_list


def find_files(scenario_dir, group):
    """Auto-detect SC_*.npy, FC_*.npy, metadata*.csv for a group in a scenario dir."""
    d = os.path.join(scenario_dir, group)
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Directory not found: {d}")
    files = os.listdir(d)
    sc   = next(f for f in files if f.startswith("SC_") and f.endswith(".npy"))
    fc   = next(f for f in files if f.startswith("FC_") and f.endswith(".npy"))
    meta = next(f for f in files if f.startswith("metadata") and f.endswith(".csv"))
    return (os.path.join(d, sc), os.path.join(d, fc), os.path.join(d, meta))


def preprocess_connectomes(SC, FC, perc, fc_norm="zscore_subject", dti_norm="minmax_subject"):
    """Zero diagonals, percentile-sparsify SC, normalize both stacks. SC/FC: (S, N, N)."""
    S, N, _ = SC.shape
    for i in range(S):
        np.fill_diagonal(SC[i], 0)
        np.fill_diagonal(FC[i], 0)
    for i in range(S):
        pos = SC[i][SC[i] > 0]
        if len(pos):
            SC[i][SC[i] < np.percentile(pos, perc)] = 0
    SC_list = list(normalize_dti(list(SC), method=dti_norm))
    FC_list = list(normalize_fc(list(FC), method=fc_norm))
    return np.stack(SC_list), np.stack(FC_list), N


def load_group(sc_path, fc_path, perc, fc_norm="zscore_subject", dti_norm="minmax_subject"):
    """Load a scenario/group's SC+FC .npy stacks, percentile-sparsify SC, normalize both."""
    SC = np.load(sc_path).astype(np.float64)
    FC = np.load(fc_path).astype(np.float64)
    return preprocess_connectomes(SC, FC, perc, fc_norm=fc_norm, dti_norm=dti_norm)


def _has_group_subfolders(scenario_dir):
    """True if scenario_dir uses the legacy layout: one subfolder per group, each
    holding its own SC_*.npy/FC_*.npy (e.g. easy/CN/SC_*.npy, easy/AD/SC_*.npy)."""
    if not os.path.isdir(scenario_dir):
        return False
    for d in os.listdir(scenario_dir):
        sub = os.path.join(scenario_dir, d)
        if os.path.isdir(sub) and glob.glob(os.path.join(sub, "SC_*.npy")):
            return True
    return False


def find_scenario_atlas(scenario_dir):
    """Atlas CSV for a scenario dir, checking both the new (scenario_dir/atlas_*.csv)
    and legacy (scenario_dir/<group>/atlas_*.csv) layouts. Returns None if not found."""
    cand = glob.glob(os.path.join(scenario_dir, "atlas_*.csv"))
    if cand:
        return cand[0]
    for d in sorted(os.listdir(scenario_dir)) if os.path.isdir(scenario_dir) else []:
        cand = glob.glob(os.path.join(scenario_dir, d, "atlas_*.csv"))
        if cand:
            return cand[0]
    return None


def load_scenario(scenario_dir, perc, fc_norm="zscore_subject", dti_norm="minmax_subject",
                   group_col="group"):
    """
    Load + preprocess one scenario directory, auto-detecting its layout:

      - legacy: one subfolder per group, each with its own SC_*.npy/FC_*.npy
        (e.g. easy/CN/SC_*.npy, easy/AD/SC_*.npy) — group label = subfolder name.
      - single-cohort: one SC_*.npy/FC_*.npy/metadata_*.csv per scenario, group
        label read from the `group_col` column of the metadata CSV.

    Returns SC (S,N,N), FC (S,N,N), groups (S,) str array, N (int), meta (DataFrame
    or None for the legacy layout, which has no per-subject metadata here).
    """
    if _has_group_subfolders(scenario_dir):
        group_dirs = sorted(
            d for d in os.listdir(scenario_dir)
            if os.path.isdir(os.path.join(scenario_dir, d))
            and glob.glob(os.path.join(scenario_dir, d, "SC_*.npy")))
        SC_parts, FC_parts, groups = [], [], []
        for g in group_dirs:
            sc_path, fc_path, _ = find_files(scenario_dir, g)
            SC_g = np.load(sc_path).astype(np.float64)
            FC_g = np.load(fc_path).astype(np.float64)
            SC_parts.append(SC_g); FC_parts.append(FC_g)
            groups += [g] * SC_g.shape[0]
        SC = np.concatenate(SC_parts); FC = np.concatenate(FC_parts)
        SC, FC, N = preprocess_connectomes(SC, FC, perc, fc_norm=fc_norm, dti_norm=dti_norm)
        return SC, FC, np.array(groups), N, None

    sc_files = glob.glob(os.path.join(scenario_dir, "SC_*.npy"))
    fc_files = glob.glob(os.path.join(scenario_dir, "FC_*.npy"))
    meta_files = glob.glob(os.path.join(scenario_dir, "metadata*.csv"))
    if not (sc_files and fc_files and meta_files):
        raise FileNotFoundError(
            f"Nessun layout riconosciuto in {scenario_dir}: attesi sottocartelle "
            f"per gruppo con SC_*.npy, oppure SC_*.npy + FC_*.npy + metadata*.csv "
            f"direttamente nella cartella.")
    SC = np.load(sc_files[0]).astype(np.float64)
    FC = np.load(fc_files[0]).astype(np.float64)
    meta = pd.read_csv(meta_files[0])
    if group_col not in meta.columns:
        raise ValueError(f"Colonna '{group_col}' non trovata in {meta_files[0]}.")
    groups = meta[group_col].astype(str).values
    SC, FC, N = preprocess_connectomes(SC, FC, perc, fc_norm=fc_norm, dti_norm=dti_norm)
    return SC, FC, groups, N, meta


def load_anagrafica(labels_path):
    """Load the AMYPAD/EPAD/EMI subject anagrafica table (xlsx or csv)."""
    if str(labels_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(labels_path)
    else:
        df = pd.read_csv(labels_path)
    if "sub" not in df.columns or "ses" not in df.columns:
        raise ValueError("L'anagrafica deve avere colonne 'sub' e 'ses'.")
    return df
