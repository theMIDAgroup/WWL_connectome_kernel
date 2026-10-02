"""Wasserstein distances between node embeddings and the pairwise distance matrix builder.

build_D is the single canonical (parallel) implementation of the pairwise Wasserstein-1 distance matrix over a list of embeddings.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment, linprog
from scipy.spatial.distance import cdist


def wasserstein_1(X1, X2):
    """Optimal one-to-one node matching. Cost = mean euclidean distance on the match."""
    M = cdist(X1, X2, metric="euclidean")
    row, col = linear_sum_assignment(M)
    return M[row, col].mean()


def wasserstein_2(X1, X2):
    """
    Optimal transport plan via Linear Programming (scipy HiGHS).
    Cost = sqrt(sum weights * distances^2), penalizes long-range transport.

    Key difference vs W1: W1 is a rigid 1-to-1 matching, indifferent to
    distribution shape; W2 is fractional transport, sensitive to the spread
    of the embedding distribution.
    """
    N1, N2 = len(X1), len(X2)
    M = cdist(X1, X2, metric="sqeuclidean")
    c = M.flatten()

    A_eq = np.zeros((N1 + N2, N1 * N2))
    for i in range(N1):
        A_eq[i, i * N2:(i + 1) * N2] = 1
    for j in range(N2):
        A_eq[N1 + j, j::N2] = 1
    b_eq = np.concatenate([np.ones(N1) / N1, np.ones(N2) / N2])

    res = linprog(c, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * (N1 * N2), method="highs")
    return float(np.sqrt(max(res.fun, 0)))


def _w1_pair(embs, i, j):
    return i, j, wasserstein_1(embs[i], embs[j])


def build_D(embs, n_jobs=-1):
    """Parallel pairwise Wasserstein-1 distance matrix over a list of embeddings."""
    S = len(embs)
    pairs = [(i, j) for i in range(S) for j in range(i + 1, S)]
    res = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_w1_pair)(embs, i, j) for i, j in pairs)
    D = np.zeros((S, S))
    for i, j, d in res:
        D[i, j] = d
    return D + D.T


def kernel_pca_2d(K):
    """2D projection via kernel PCA on a centered precomputed kernel matrix."""
    S = K.shape[0]
    one = np.ones((S, S)) / S
    Kc = K - one @ K - K @ one + one @ K @ one
    vals, vecs = eigh(Kc)
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    pos_mask = vals > 0
    Z = np.zeros((S, 2))
    for d in range(min(2, pos_mask.sum())):
        Z[:, d] = vecs[:, d] * np.sqrt(max(vals[d], 0))
    return Z
