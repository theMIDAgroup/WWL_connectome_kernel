"""Nested cross-validation, ablation kernels, and the permutation test.

nested_cv_kernel merges wwl_benchmark.py's nested_cv_kernel (lambda
CV-calibrated per fold) and WWL_full_pipeline.py's kernel_svm_nested_cv
(lambda "1/mu", richer per-fold diagnostics incl. confusion matrix) into one
function: pass lam_method="1/mu" and return_diagnostics=True to reproduce the
latter.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse.csgraph import floyd_warshall
from scipy.stats import pearsonr, spearmanr
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .distances import build_D
from .embedding import wl_embedding, wl_embedding_fractional
from .kernels import build_K, calibrate_lam, calibrate_lam_and_C

# half-decade steps (was 5 whole-decade points 0.01..100) — finer C selection
# without changing the overall searched range
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
ALPHA_GRID = [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]


def nested_cv_kernel(D, y, n_outer, n_inner, seed, lam_method="cv", return_diagnostics=False,
                      n_jobs=-1, C_grid=C_GRID):
    """lambda calibrated on each train fold (lam_method); C tuned by inner CV
    over C_grid (default C_GRID)."""
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)

    per_fold = []
    y_true_all, y_pred_all = [], []

    for fold, (tr, te) in enumerate(outer.split(D, y)):
        D_tr, D_te = D[np.ix_(tr, tr)], D[np.ix_(te, tr)]
        y_tr, y_te = y[tr], y[te]

        if lam_method == "cv":
            # joint (lambda, C) grid search — see calibrate_lam_and_C docstring
            # for why this beats calibrating lambda first with C pinned at 1.0
            lam, best_C = calibrate_lam_and_C(D_tr, y_tr, C_grid, seed=seed, n_jobs=n_jobs)
        else:
            lam = calibrate_lam(D_tr, y_tr, method=lam_method)
            K_tr_tmp = build_K(D_tr, lam)
            best_C, best_acc = C_grid[0], -1
            for C in C_grid:
                accs = []
                for itr, ite in inner.split(K_tr_tmp, y_tr):
                    svm = SVC(kernel="precomputed", C=C, class_weight="balanced")
                    svm.fit(K_tr_tmp[np.ix_(itr, itr)], y_tr[itr])
                    accs.append(balanced_accuracy_score(
                        y_tr[ite], svm.predict(K_tr_tmp[np.ix_(ite, itr)])))
                if np.mean(accs) > best_acc:
                    best_acc, best_C = np.mean(accs), C

        K_tr, K_te = build_K(D_tr, lam), build_K(D_te, lam)
        svm = SVC(kernel="precomputed", C=best_C, class_weight="balanced")
        svm.fit(K_tr, y_tr)
        y_pred = svm.predict(K_te)
        acc  = float((y_pred == y_te).mean())
        bacc = balanced_accuracy_score(y_te, y_pred)
        y_true_all.extend(y_te); y_pred_all.extend(y_pred)
        per_fold.append({"fold": fold + 1, "acc": acc, "bacc": bacc,
                          "lambda": lam, "best_C": best_C,
                          "n_train": len(tr), "n_test": len(te)})

    baccs = [r["bacc"] for r in per_fold]
    result = (float(np.mean(baccs)), float(np.std(baccs)))
    if not return_diagnostics:
        return result

    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    diagnostics = {
        "per_fold":    pd.DataFrame(per_fold),
        "mean_acc":    np.mean([r["acc"] for r in per_fold]),
        "std_acc":     np.std([r["acc"] for r in per_fold]),
        "mean_bacc":   result[0],
        "std_bacc":    result[1],
        "mean_lambda": np.mean([r["lambda"] for r in per_fold]),
        "y_true": y_true_all, "y_pred": y_pred_all,
        "cm": confusion_matrix(y_true_all, y_pred_all),
    }
    return result, diagnostics


def nested_cv_kernel_ordinal(D, y, n_outer, n_inner, seed, lam_method="cv", C_grid=C_GRID):
    """
    Ordinal-aware alternative to nested_cv_kernel for classes with a natural
    order (e.g. CN < MCI < AD), via Frank & Hall binary decomposition: train
    K-1 binary "is severity beyond threshold k?" SVCs on the SAME calibrated
    kernel (rather than one-vs-one multiclass SVC, which treats "CN vs AD"
    and "CN vs MCI" as unrelated problems and discards the ordering), and
    combine by counting how many thresholds each test subject is predicted
    to have crossed. Uses the same calibrate_lam_and_C joint search as
    nested_cv_kernel on the original multiclass y, so any accuracy
    difference from nested_cv_kernel reflects the classification strategy,
    not a different kernel calibration.

    y must be integer labels 0..K-1, in increasing order of severity.
    Returns (mean_bacc, std_bacc) — same shape as nested_cv_kernel.
    """
    classes = np.unique(y)
    K = len(classes)
    if K < 3:
        raise ValueError("nested_cv_kernel_ordinal needs >=3 ordered classes "
                          "(for 2 classes it's identical to nested_cv_kernel).")

    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    baccs = []

    for tr, te in outer.split(D, y):
        D_tr, D_te = D[np.ix_(tr, tr)], D[np.ix_(te, tr)]
        y_tr, y_te = y[tr], y[te]

        if lam_method == "cv":
            lam, C = calibrate_lam_and_C(D_tr, y_tr, C_grid, seed=seed)
        else:
            lam = calibrate_lam(D_tr, y_tr, method=lam_method)
            C = 1.0

        K_tr, K_te = build_K(D_tr, lam), build_K(D_te, lam)

        crossed = np.zeros(len(te), dtype=int)
        for k in classes[:-1]:
            y_bin = (y_tr > k).astype(int)
            if len(np.unique(y_bin)) < 2:
                continue  # degenerate threshold, shouldn't happen with balanced groups
            svm = SVC(kernel="precomputed", C=C, class_weight="balanced")
            svm.fit(K_tr, y_bin)
            crossed += svm.predict(K_te).astype(int)

        y_pred = np.clip(crossed, 0, K - 1)
        baccs.append(balanced_accuracy_score(y_te, y_pred))

    return float(np.mean(baccs)), float(np.std(baccs))


def calibrate_lam_and_alpha_regression(D_train, y_train, alpha_grid=ALPHA_GRID,
                                        n_splits=3, seed=0, lam_grid=None, n_jobs=-1):
    """
    Regression analogue of kernels.calibrate_lam_and_C: joint grid search over
    (lambda for the Laplacian kernel, ridge alpha for KernelRidge) via inner
    CV, scored by negative MSE. Joint rather than sequential for the same
    reason as the classification version — lambda's effect on kernel scale
    interacts with the regularization strength that is optimal for it.
    """
    vals = D_train[np.triu_indices(len(D_train), k=1)]
    lam_mu = 1.0 / (vals.mean() + 1e-9)
    if lam_grid is None:
        lam_grid = np.logspace(np.log10(lam_mu * 0.05), np.log10(lam_mu * 50), 15)

    inner = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(inner.split(D_train))

    def _score(lam, alpha):
        K = build_K(D_train, lam)
        errs = []
        for itr, ite in splits:
            model = KernelRidge(kernel="precomputed", alpha=alpha)
            model.fit(K[np.ix_(itr, itr)], y_train[itr])
            pred = model.predict(K[np.ix_(ite, itr)])
            errs.append(-np.mean((pred - y_train[ite]) ** 2))
        return float(np.mean(errs))

    combos = [(lam, a) for lam in lam_grid for a in alpha_grid]
    scores = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_score)(lam, a) for lam, a in combos)
    best_lam, best_alpha = combos[int(np.argmax(scores))]
    return best_lam, best_alpha


def nested_cv_kernel_regression(D, y, n_outer, n_inner, seed, alpha_grid=ALPHA_GRID,
                                 return_diagnostics=False, n_jobs=-1):
    """
    Continuous-target counterpart to nested_cv_kernel: same precomputed
    Laplacian kernel K = exp(-lambda*D), but KernelRidge instead of SVC, and
    plain KFold instead of StratifiedKFold (y is continuous, nothing to
    stratify on). lambda and ridge alpha are jointly calibrated per outer
    fold via calibrate_lam_and_alpha_regression, mirroring how
    nested_cv_kernel calibrates (lambda, C).

    Returns a dict with out-of-fold y_true/y_pred plus overall R2, Pearson r
    (and p), Spearman rho (and p) — Spearman is reported alongside Pearson
    because several AMYPAD-PNHS biomarkers (Centiloidi, CSF ptau) are
    right-skewed, where a monotonic-but-nonlinear dose-response would show
    up in Spearman but be understated by R2/Pearson.
    """
    outer = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    per_fold = []
    y_true_all, y_pred_all = [], []

    for fold, (tr, te) in enumerate(outer.split(D)):
        D_tr, D_te = D[np.ix_(tr, tr)], D[np.ix_(te, tr)]
        y_tr, y_te = y[tr], y[te]

        lam, alpha = calibrate_lam_and_alpha_regression(
            D_tr, y_tr, alpha_grid=alpha_grid, n_splits=n_inner, seed=seed, n_jobs=n_jobs)

        K_tr, K_te = build_K(D_tr, lam), build_K(D_te, lam)
        model = KernelRidge(kernel="precomputed", alpha=alpha)
        model.fit(K_tr, y_tr)
        y_pred = model.predict(K_te)

        y_true_all.extend(y_te); y_pred_all.extend(y_pred)
        per_fold.append({"fold": fold + 1, "lambda": lam, "alpha": alpha,
                          "n_train": len(tr), "n_test": len(te)})

    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    ss_res = np.sum((y_true_all - y_pred_all) ** 2)
    ss_tot = np.sum((y_true_all - y_true_all.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    pearson_r, pearson_p = pearsonr(y_true_all, y_pred_all)
    spearman_r, spearman_p = spearmanr(y_true_all, y_pred_all)

    result = {
        "r2": r2, "pearson_r": float(pearson_r), "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p": float(spearman_p),
        "y_true": y_true_all, "y_pred": y_pred_all,
    }
    if return_diagnostics:
        result["per_fold"] = pd.DataFrame(per_fold)
    return result


def permutation_test_regression(D, y, n_perm, n_outer, seed, alpha_grid=ALPHA_GRID, n_jobs=-1):
    """Regression analogue of permutation_test: permute y, recompute nested-CV
    R2 each time -> null distribution, to test whether the observed R2 (or
    Pearson/Spearman r) beats chance given the sample size."""
    rng = np.random.RandomState(seed)

    def _one_perm(perm_seed):
        rng_p = np.random.RandomState(perm_seed)
        y_perm = rng_p.permutation(y)
        res = nested_cv_kernel_regression(D, y_perm, n_outer, n_inner=3, seed=seed,
                                           alpha_grid=alpha_grid, n_jobs=1)
        return res["r2"]

    perm_seeds = rng.randint(0, 1_000_000, n_perm)
    null_dist = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_perm)(s) for s in perm_seeds)
    return np.array(null_dist)


def nested_cv_flat(SC, FC, y, n_outer, n_inner, seed, return_diagnostics=False,
                    C_grid=(0.001, 0.01, 0.1, 1, 10)):
    """Baseline: upper-triangle SC+FC concatenated, StandardScaler + linear SVM."""
    idx = np.triu_indices(SC.shape[1], k=1)
    X = np.hstack([SC[:, idx[0], idx[1]], FC[:, idx[0], idx[1]]])
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
    baccs = []
    y_true_all, y_pred_all = [], []
    for tr, te in outer.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        best_C, best_acc = C_grid[0], -1
        for C in C_grid:
            accs = []
            for itr, ite in inner.split(X_tr_s, y_tr):
                svm = SVC(kernel="linear", C=C, class_weight="balanced")
                svm.fit(X_tr_s[itr], y_tr[itr])
                accs.append(balanced_accuracy_score(
                    y_tr[ite], svm.predict(X_tr_s[ite])))
            if np.mean(accs) > best_acc:
                best_acc, best_C = np.mean(accs), C
        svm = SVC(kernel="linear", C=best_C, class_weight="balanced")
        svm.fit(X_tr_s, y_tr)
        y_pred = svm.predict(X_te_s)
        baccs.append(balanced_accuracy_score(y_te, y_pred))
        y_true_all.extend(y_te); y_pred_all.extend(y_pred)
    result = (float(np.mean(baccs)), float(np.std(baccs)))
    if not return_diagnostics:
        return result
    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    return result, {"y_true": y_true_all, "y_pred": y_pred_all,
                     "cm": confusion_matrix(y_true_all, y_pred_all)}


def wl_subtree_kernel(embs_all):
    """Approximate WL subtree kernel: inner product between row-normalized flattened embeddings."""
    X = np.array([e.ravel() for e in embs_all])
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-9
    X_n = X / norms
    return X_n @ X_n.T


def nested_cv_subtree(K, y, n_outer, n_inner, seed, return_diagnostics=False, C_grid=C_GRID):
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
    baccs = []
    y_true_all, y_pred_all = [], []
    for tr, te in outer.split(K, y):
        K_tr, K_te = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
        y_tr, y_te = y[tr], y[te]
        best_C, best_acc = C_grid[0], -1
        for C in C_grid:
            accs = []
            for itr, ite in inner.split(K_tr, y_tr):
                svm = SVC(kernel="precomputed", C=C, class_weight="balanced")
                svm.fit(K_tr[np.ix_(itr, itr)], y_tr[itr])
                accs.append(balanced_accuracy_score(
                    y_tr[ite], svm.predict(K_tr[np.ix_(ite, itr)])))
            if np.mean(accs) > best_acc:
                best_acc, best_C = np.mean(accs), C
        svm = SVC(kernel="precomputed", C=best_C, class_weight="balanced")
        svm.fit(K_tr, y_tr)
        y_pred = svm.predict(K_te)
        baccs.append(balanced_accuracy_score(y_te, y_pred))
        y_true_all.extend(y_te); y_pred_all.extend(y_pred)
    result = (float(np.mean(baccs)), float(np.std(baccs)))
    if not return_diagnostics:
        return result
    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    return result, {"y_true": y_true_all, "y_pred": y_pred_all,
                     "cm": confusion_matrix(y_true_all, y_pred_all)}


def lambda_sensitivity_curve(D, y, lam_mu=None, n_splits=3, seed=42, n_points=40, span=(0.02, 100)):
    """Balanced accuracy (n_splits-fold CV, C=1) across a log-spaced lambda grid around lam_mu."""
    if lam_mu is None:
        lam_mu = 1.0 / (D[D > 0].mean() + 1e-9)
    lam_grid = np.logspace(np.log10(lam_mu * span[0]), np.log10(lam_mu * span[1]), n_points)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for lam in lam_grid:
        K = build_K(D, lam)
        fold_accs = []
        for tr, te in cv.split(K, y):
            svm = SVC(kernel="precomputed", C=1.0, class_weight="balanced")
            svm.fit(K[np.ix_(tr, tr)], y[tr])
            fold_accs.append(balanced_accuracy_score(y[te], svm.predict(K[np.ix_(te, tr)])))
        accs.append(np.mean(fold_accs))
    return lam_grid, np.array(accs)


def permutation_test(D, y, n_perm, n_outer, seed, n_jobs=-1, lam_method="1/mu", C=1.0):
    """Permute labels n_perm times, recompute balanced accuracy each time -> null distribution."""
    rng = np.random.RandomState(seed)
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)

    def _one_perm(perm_seed):
        rng_p = np.random.RandomState(perm_seed)
        y_perm = rng_p.permutation(y)
        baccs = []
        for tr, te in outer.split(D, y_perm):
            D_tr, D_te = D[np.ix_(tr, tr)], D[np.ix_(te, tr)]
            lam = calibrate_lam(D_tr, method=lam_method)
            K_tr, K_te = build_K(D_tr, lam), build_K(D_te, lam)
            svm = SVC(kernel="precomputed", C=C, class_weight="balanced")
            svm.fit(K_tr, y_perm[tr])
            baccs.append(balanced_accuracy_score(y_perm[te], svm.predict(K_te)))
        return float(np.mean(baccs))

    perm_seeds = rng.randint(0, 1_000_000, n_perm)
    null_dist = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_one_perm)(s) for s in perm_seeds)
    return np.array(null_dist)


def select_best_h(FC, SC, y, h_values, n_outer, n_inner, seed, n_jobs=-1):
    """
    Try several WL-propagation depths H (e.g. [1, 2, 3]): for each, recompute
    the WL embeddings + W1 distance matrix, then run the SAME nested_cv_kernel
    (which already cross-validates lambda and C per outer fold) to get an
    honest nested-CV balanced accuracy estimate. H was previously a fixed
    constant everywhere in the pipeline — this makes it a tuned hyperparameter
    like lambda/C already were.

    Embeddings depend only on each subject's own (FC, SC, H), never on labels
    or other subjects, so recomputing D per H introduces no leakage; the one
    caveat is the usual one for "pick the best of several honest CV
    estimates" — comparing multiple H values has a mild optimistic selection
    bias on top of (not instead of) each individual estimate being unbiased.

    FC, SC    : (S, N, N) arrays, one entry per subject
    h_values  : iterable of int

    Returns (results, best_h) where results = {h: {"mean_bacc", "std_bacc", "D", "embs"}}.
    """
    results = {}
    for h in h_values:
        embs = [wl_embedding(FC[i], SC[i].copy(), h) for i in range(len(FC))]
        D = build_D(embs, n_jobs=n_jobs)
        mean_bacc, std_bacc = nested_cv_kernel(D, y, n_outer, n_inner, seed, n_jobs=n_jobs)
        results[h] = {"mean_bacc": mean_bacc, "std_bacc": std_bacc, "D": D, "embs": embs}

    best_h = max(results, key=lambda h: results[h]["mean_bacc"])
    return results, best_h


def select_best_h_alpha(FC, SC, y, h_values, alpha_values, n_outer, n_inner, seed,
                         beta=1.0, n_jobs=-1):
    """
    Fractional-propagation analogue of select_best_h: for each (H, alpha)
    combination, recompute the WL embeddings with the regularized fractional
    structural propagation (embedding.wl_embedding_fractional, Filippo &
    Mazza 2026) instead of the direct structural adjacency, then score with
    the same nested_cv_kernel used for select_best_h's plain-H results — same
    classifier, same CV splits, same lambda/C calibration — so the two are
    directly comparable.

    alpha in (0, 1): smaller alpha -> more non-local (closer to the
    graph-independent limit of Lemma 3.3); alpha -> 1 recovers plain DTI
    propagation (Filippo & Mazza, Lemma 3.2), i.e. select_best_h's embedding.

    FC, SC       : (S, N, N) arrays, one entry per subject
    h_values     : iterable of int
    alpha_values : iterable of float in (0, 1)
    beta         : float >= 1, non-local scaling shared by all combinations
                   (Filippo & Mazza Def. 4.1)

    Returns (results, best) where results = {(h, alpha): {"mean_bacc",
    "std_bacc", "D", "embs"}} and best is the best-scoring (h, alpha) key.
    """
    results = {}
    for h in h_values:
        for alpha in alpha_values:
            embs = [wl_embedding_fractional(FC[i], SC[i].copy(), h, alpha, beta=beta)
                    for i in range(len(FC))]
            D = build_D(embs, n_jobs=n_jobs)
            mean_bacc, std_bacc = nested_cv_kernel(D, y, n_outer, n_inner, seed, n_jobs=n_jobs)
            results[(h, alpha)] = {"mean_bacc": mean_bacc, "std_bacc": std_bacc, "D": D, "embs": embs}

    best = max(results, key=lambda k: results[k]["mean_bacc"])
    return results, best


# ─────────────────────────────────────────────────────────────────────────────
# EXTERNAL-METHOD BASELINES — genuinely different classification approaches
# from the literature, not ablations of the WWL pipeline itself. Used to give
# WWL an honest external comparison rather than only internal ablations.
# ─────────────────────────────────────────────────────────────────────────────
def graph_theory_features(FC, rsn_ids):
    """
    Classic network-neuroscience feature set per subject, extracted from FC:
      - node strength (weighted degree, off-diagonal sum)            (N features)
      - weighted clustering coefficient (Onnela et al. 2005)         (N features)
      - modularity Q (Newman), using the KNOWN RSN partition          (1 feature)
      - density (fraction of nonzero edges)                          (1 feature)
    Fully vectorized (no networkx), fast enough for N~100 region graphs
    across hundreds of subjects. Returns (S, 2N+2).
    """
    S, N, _ = FC.shape
    feats = np.zeros((S, 2 * N + 2))
    rsn_ids = np.asarray(rsn_ids)
    same_rsn = (rsn_ids[:, None] == rsn_ids[None, :]).astype(float)

    for s in range(S):
        W = np.abs(FC[s]).astype(float).copy()
        np.fill_diagonal(W, 0)
        A = (W > 0).astype(float)
        deg = A.sum(axis=1)
        strength = W.sum(axis=1)

        # Onnela et al. 2005 weighted clustering coefficient, vectorized via
        # the cube-root-weighted adjacency matrix cubed (closed weighted triplets)
        Wc = np.cbrt(W)
        triangles = np.diag(Wc @ Wc @ Wc)
        denom = deg * (deg - 1)
        denom[denom == 0] = 1
        clustering = triangles / denom

        m = W.sum() / 2
        if m > 0:
            B = W - np.outer(strength, strength) / (2 * m)
            Q = (B * same_rsn).sum() / (2 * m)
        else:
            Q = 0.0
        density = A.sum() / (N * (N - 1))

        feats[s] = np.concatenate([strength, clustering, [Q, density]])
    return feats


def nested_cv_graphtheory(FC, y, rsn_ids, n_outer, n_inner, seed, return_diagnostics=False,
                           C_grid=C_GRID):
    """
    Classic network-neuroscience baseline: graph_theory_features + linear SVM,
    same nested-CV/C-grid structure as nested_cv_flat. A genuinely different
    approach from WWL (hand-crafted graph-theoretic summary statistics
    instead of a learned graph kernel), not an ablation of it.
    """
    X = graph_theory_features(FC, rsn_ids)
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
    baccs = []
    y_true_all, y_pred_all = [], []
    for tr, te in outer.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        best_C, best_acc = C_grid[0], -1
        for C in C_grid:
            accs = []
            for itr, ite in inner.split(X_tr_s, y_tr):
                svm = SVC(kernel="linear", C=C, class_weight="balanced")
                svm.fit(X_tr_s[itr], y_tr[itr])
                accs.append(balanced_accuracy_score(y_tr[ite], svm.predict(X_tr_s[ite])))
            if np.mean(accs) > best_acc:
                best_acc, best_C = np.mean(accs), C
        svm = SVC(kernel="linear", C=best_C, class_weight="balanced")
        svm.fit(X_tr_s, y_tr)
        y_pred = svm.predict(X_te_s)
        baccs.append(balanced_accuracy_score(y_te, y_pred))
        y_true_all.extend(y_te); y_pred_all.extend(y_pred)
    result = (float(np.mean(baccs)), float(np.std(baccs)))
    if not return_diagnostics:
        return result
    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    return result, {"y_true": y_true_all, "y_pred": y_pred_all,
                     "cm": confusion_matrix(y_true_all, y_pred_all)}


def shortest_path_histograms(FC, n_bins=30, eps=1e-6):
    """
    Per-subject histogram of all-pairs shortest-path lengths on a weighted
    graph built from FC (distance = 1/|correlation|, so strongly connected
    region pairs are "close"). Tractable histogram simplification of the
    Borgwardt & Kriegel (2005) shortest-path kernel: the full pairwise
    edge-length matching kernel is O(N^4) per graph PAIR, infeasible at
    N~100 regions; the shortest-path-length distribution captured here is
    O(N^3) per SUBJECT (Floyd-Warshall) and still a genuinely different
    graph descriptor from WL propagation (global path structure rather than
    local iterative neighbourhood aggregation).
    """
    S, N, _ = FC.shape
    all_finite = []
    for s in range(S):
        W = np.abs(FC[s]).astype(float).copy()
        np.fill_diagonal(W, 0)
        with np.errstate(divide="ignore"):
            dist = np.where(W > eps, 1.0 / W, np.inf)
        np.fill_diagonal(dist, 0)
        sp = floyd_warshall(dist, directed=False)
        finite = sp[np.isfinite(sp) & (sp > 0)]
        all_finite.append(finite)

    all_vals = np.concatenate(all_finite) if all_finite else np.array([0.0, 1.0])
    lo, hi = np.percentile(all_vals, [1, 99])
    bins = np.linspace(max(lo, 1e-9), max(hi, lo + 1e-9), n_bins + 1)

    hists = np.array([np.histogram(f, bins=bins, density=True)[0] for f in all_finite])
    return hists


def nested_cv_shortest_path(FC, y, n_outer, n_inner, seed, n_bins=30, return_diagnostics=False,
                             C_grid=C_GRID, gamma_grid=("scale", "auto", 0.01, 0.1, 1.0)):
    """
    Shortest-path kernel baseline (Borgwardt & Kriegel 2005, histogram
    simplification — see shortest_path_histograms): per-subject
    shortest-path-length histogram, classified via RBF-SVM (C and gamma
    grid search). A graph kernel mechanism genuinely different from WWL's
    WL-propagation + optimal-transport distance.
    """
    X = shortest_path_histograms(FC, n_bins=n_bins)
    outer = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1)
    baccs = []
    y_true_all, y_pred_all = [], []
    for tr, te in outer.split(X, y):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        best_params, best_acc = (C_grid[0], gamma_grid[0]), -1
        for C in C_grid:
            for gamma in gamma_grid:
                accs = []
                for itr, ite in inner.split(X_tr_s, y_tr):
                    svm = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")
                    svm.fit(X_tr_s[itr], y_tr[itr])
                    accs.append(balanced_accuracy_score(y_tr[ite], svm.predict(X_tr_s[ite])))
                if np.mean(accs) > best_acc:
                    best_acc, best_params = np.mean(accs), (C, gamma)
        C, gamma = best_params
        svm = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")
        svm.fit(X_tr_s, y_tr)
        y_pred = svm.predict(X_te_s)
        baccs.append(balanced_accuracy_score(y_te, y_pred))
        y_true_all.extend(y_te); y_pred_all.extend(y_pred)
    result = (float(np.mean(baccs)), float(np.std(baccs)))
    if not return_diagnostics:
        return result
    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    return result, {"y_true": y_true_all, "y_pred": y_pred_all,
                     "cm": confusion_matrix(y_true_all, y_pred_all)}
