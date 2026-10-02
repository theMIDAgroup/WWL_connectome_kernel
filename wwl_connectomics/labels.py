"""Real-data subject/session selection and anagrafica target-label extraction.
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .io import normalize_dti, normalize_fc


def _session_to_int(session_label):
    m = re.search(r"(\d+)", str(session_label))
    return int(m.group(1)) if m else 10 ** 9


def select_first_session(pairs):
    """
    pairs[session_label][subject] = {"fmri":..., "dti":..., "study":...,
                                      "ses_raw":..., "visit_date":..., ...}

    Returns dict subject -> record, from the lowest canonical session in
    which the subject appears.
    """
    subj_best_session = {}
    for session_label, subj_dict in pairs.items():
        s_int = _session_to_int(session_label)
        for subject in subj_dict:
            if subject not in subj_best_session or s_int < subj_best_session[subject][0]:
                subj_best_session[subject] = (s_int, session_label)

    first_session_data = {}
    for subject, (s_int, session_label) in subj_best_session.items():
        first_session_data[subject] = dict(pairs[session_label][subject])
        first_session_data[subject]["session_used"] = session_label
    return first_session_data


def build_arrays(first_session_data, perc, fc_norm, dti_norm):
    """Same preprocessing schema as io.load_group, for a subject->record dict."""
    subjects = sorted(first_session_data.keys())
    SC = np.stack([np.asarray(first_session_data[s]["dti"], dtype=np.float64)
                   for s in subjects])
    FC = np.stack([np.asarray(first_session_data[s]["fmri"], dtype=np.float64)
                   for s in subjects])
    studies  = [first_session_data[s].get("study") for s in subjects]
    sessions = [first_session_data[s].get("session_used") for s in subjects]
    ses_raw  = [first_session_data[s].get("ses_raw") for s in subjects]

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
    SC = np.stack(SC_list)
    FC = np.stack(FC_list)

    return subjects, SC, FC, N, studies, sessions, ses_raw


def build_target_labels(anagrafica_df, target_col, subjects, ses_raw):
    """
    Extract target_col from the anagrafica for each (subject, ses_raw).
    Non-binary columns are dichotomized by median.

    Returns: kept_subjects, y (int array), info dict (class mapping, threshold, ...)
    """
    if target_col not in anagrafica_df.columns:
        raise ValueError(f"Colonna '{target_col}' non trovata nell'anagrafica.")

    lookup = anagrafica_df.set_index(["sub", "ses"])[target_col]

    raw_vals, kept_subjects = [], []
    missing = 0
    for subj, ses in zip(subjects, ses_raw):
        key = (subj, ses)
        if key in lookup.index:
            val = lookup.loc[key]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
        else:
            val = np.nan
        if pd.isna(val):
            missing += 1
            continue
        raw_vals.append(val)
        kept_subjects.append(subj)

    if missing:
        print(f"    {missing} soggetti esclusi per '{target_col}' "
              f"(match sub+ses non trovato o valore mancante)")

    raw_series = pd.Series(raw_vals)
    n_unique = raw_series.nunique()

    info = {"target": target_col}
    if n_unique <= 2:
        le = LabelEncoder()
        y = le.fit_transform(raw_series.astype(str))
        info["mode"] = "categorical_as_is"
        info["classes"] = dict(zip(le.classes_, range(len(le.classes_))))
    else:
        median = raw_series.median()
        y = (raw_series > median).astype(int).values
        info["mode"] = "median_split"
        info["median"] = float(median)
        info["classes"] = {f"<= {median}": 0, f"> {median}": 1}
        print(f"    '{target_col}' non binaria ({n_unique} valori distinti): "
              f"dicotomizzata per mediana = {median}")

    print(f"    Classi per '{target_col}': {info['classes']}  "
          f"(N={len(kept_subjects)})")
    return kept_subjects, np.asarray(y), info
