"""Per-subject, per-region embedding shift (||a^H(r) - a^0(r)||) and group statistics.

compute_shifts used to be reimplemented identically in wwl_benchmark.py
(inline in run_scenario), make_scenario_figures.py and wwl_real_node_shift.py.
compute_region_shift_stats centralizes the FDR-corrected Mann-Whitney block
duplicated between wwl_benchmark.plot_population_shift and
wwl_real_node_shift.plot_node_shift_ranking.
"""

import numpy as np
from scipy.stats import mannwhitneyu


def compute_shifts(embs, N):
    """shift[i, r] = ||a^H(r) - a^0(r)|| for subject i, region r. embs: list of (N, N*(H+1))."""
    return np.array([
        np.linalg.norm(e[:, -N:] - e[:, :N], axis=1)
        for e in embs
    ])


def compute_region_shift_stats(shifts_0, shifts_1):
    """
    Per-region shift statistics between two groups (shifts_0/1: (S, N) arrays).
    delta = mean_1 - mean_0 (positive => more shift in group 1).
    p_mw  : one-sided Mann-Whitney (group 1 > group 0), per region
    p_fdr : Benjamini-Hochberg corrected p_mw
    """
    N = shifts_0.shape[1]
    mu_0, sd_0 = shifts_0.mean(axis=0), shifts_0.std(axis=0)
    mu_1, sd_1 = shifts_1.mean(axis=0), shifts_1.std(axis=0)
    delta = mu_1 - mu_0

    p_vals = np.array([
        mannwhitneyu(shifts_1[:, r], shifts_0[:, r], alternative="greater").pvalue
        for r in range(N)
    ])
    order = np.argsort(p_vals)
    p_fdr = np.empty(N)
    for rank, idx in enumerate(order):
        p_fdr[idx] = min(p_vals[idx] * N / (rank + 1), 1.0)
    significant = p_fdr < 0.05

    return {
        "mu_0": mu_0, "sd_0": sd_0, "mu_1": mu_1, "sd_1": sd_1,
        "delta": delta, "p_mw": p_vals, "p_fdr": p_fdr, "significant": significant,
    }


def region_shift_dataframe(stats, region_names, rsn_labels=None, label_0="group0", label_1="group1"):
    """Tidy DataFrame from compute_region_shift_stats output, sorted by |delta| desc... actually delta desc."""
    import pandas as pd

    data = {
        "region": region_names,
        f"mean_shift_{label_0}": stats["mu_0"],
        f"std_shift_{label_0}":  stats["sd_0"],
        f"mean_shift_{label_1}": stats["mu_1"],
        f"std_shift_{label_1}":  stats["sd_1"],
        "delta_shift": stats["delta"],
        "p_mw": stats["p_mw"], "p_fdr": stats["p_fdr"],
        "significant": stats["significant"],
    }
    if rsn_labels is not None:
        data["rsn"] = rsn_labels
    return pd.DataFrame(data).sort_values("delta_shift", ascending=False)
