"""Laplacian kernel K = exp(-lambda * D) and lambda calibration strategies.

calibrate_lam merges the 3 variants that used to live independently in
wwl_benchmark.py ("1/mu" and "cv"), make_scenario_figures.py
(calibrate_lam_fisher) and WWL_full_pipeline.py (calibrate_lambda, "1/mu" only).
"""

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

METHOD_ORDER = ["WWL", "SC-only", "FC-only", "WL-subtree", "Flat-SVM",
                 "GraphTheory", "ShortestPath"]
METHOD_LABELS = {
    "WWL":          "WWL\n(proposed)",
    "SC-only":      "SC-only\nkernel",
    "FC-only":      "FC-only\nkernel",
    "WL-subtree":   "WL subtree\n(Shervashidze '11)",
    "Flat-SVM":     "Flat\nSVM",
    "GraphTheory":  "Graph theory\n+ SVM",
    "ShortestPath": "Shortest-path\nkernel",
}


def calibrate_lam(D_train, y_train=None, method="cv"):
    """
    Calibrate lambda for the Laplacian kernel K = exp(-lambda * D).

    method="1/mu" : lambda = 1 / mean(D_train)  [fast, no labels needed]
    method="cv"   : lambda chosen by 3-fold CV on training distances [better]
    """
    vals = D_train[np.triu_indices(len(D_train), k=1)]
    lam_mu = 1.0 / (vals.mean() + 1e-9)

    if method == "1/mu" or y_train is None:
        return lam_mu

    lam_grid = np.logspace(np.log10(lam_mu * 0.05), np.log10(lam_mu * 50), 30)
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    best_lam, best_acc = lam_mu, -1
    for lam in lam_grid:
        K = np.exp(-lam * D_train)
        accs = []
        for itr, ite in inner.split(K, y_train):
            svm = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            svm.fit(K[np.ix_(itr, itr)], y_train[itr])
            accs.append(balanced_accuracy_score(
                y_train[ite], svm.predict(K[np.ix_(ite, itr)])))
        mu = float(np.mean(accs))
        if mu > best_acc:
            best_acc, best_lam = mu, lam
    return best_lam


def calibrate_lam_and_C(D_train, y_train, C_grid, n_splits=3, seed=0, lam_grid=None):
    """
    Jointly grid-search (lambda, C) via inner CV, instead of calibrating
    lambda first with C pinned at 1.0 (calibrate_lam's "cv" method) and only
    then searching C with that lambda fixed — the two interact (a larger
    lambda makes K peakier, which shifts the SVM's effective margin and thus
    the optimal C), so a sequential search can miss the joint optimum.

    Returns (best_lam, best_C).
    """
    vals = D_train[np.triu_indices(len(D_train), k=1)]
    lam_mu = 1.0 / (vals.mean() + 1e-9)
    if lam_grid is None:
        lam_grid = np.logspace(np.log10(lam_mu * 0.05), np.log10(lam_mu * 50), 15)

    inner = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_lam, best_C, best_acc = lam_mu, C_grid[len(C_grid) // 2], -1
    for lam in lam_grid:
        K = np.exp(-lam * D_train)
        for C in C_grid:
            accs = []
            for itr, ite in inner.split(K, y_train):
                svm = SVC(kernel="precomputed", C=C, class_weight="balanced")
                svm.fit(K[np.ix_(itr, itr)], y_train[itr])
                accs.append(balanced_accuracy_score(
                    y_train[ite], svm.predict(K[np.ix_(ite, itr)])))
            mu = float(np.mean(accs))
            if mu > best_acc:
                best_acc, best_lam, best_C = mu, lam, C
    return best_lam, best_C


def calibrate_lam_fisher(D, groups):
    """
    Lambda that maximizes Fisher separability (intra- vs inter-group kernel
    similarity) rather than classification accuracy.

    groups: array-like of group labels, one per row/col of D (any number of
    distinct values >= 2; only the intra/inter split matters).
    Returns (lam_fisher, lam_mu).
    """
    groups = np.asarray(groups)
    S = len(D)
    lam_mu = 1.0 / (D[D > 0].mean() + 1e-9)

    intra, inter = [], []
    for i in range(S):
        for j in range(i + 1, S):
            (intra if groups[i] == groups[j] else inter).append(D[i, j])
    intra, inter = np.array(intra), np.array(inter)

    def neg_fisher(lam):
        ki, ke = np.exp(-lam * intra), np.exp(-lam * inter)
        sep = ki.mean() - ke.mean()
        var = (ki.var() + ke.var()) / 2 + 1e-9
        return -sep ** 2 / var

    res = minimize_scalar(neg_fisher, bounds=(1e-4, lam_mu * 50), method="bounded")
    return res.x, lam_mu


def build_K(D, lam):
    return np.exp(-lam * D)
