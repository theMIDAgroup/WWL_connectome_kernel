"""Weisfeiler-Lehman continuous embedding propagation (eq. 5, Togninalli et al. 2019)."""

import numpy as np

from .fractional import regularized_fractional_weight


def wl_embedding(FC, DTI, H=2):
    """
    FC  : (N, N) — each row is the node's initial attribute vector
    DTI : (N, N) — edge weights used for propagation
    H   : number of WL iterations
    → (N, N*(H+1)) concatenated embedding
    """
    N = FC.shape[0]
    np.fill_diagonal(DTI, 0)

    deg = DTI.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1
    W = DTI / deg

    a = FC.copy()
    layers = [a.copy()]

    for _ in range(H):
        aux = 0.5 * (a + W @ a)
        layers.append(aux.copy())
        a = aux

    return np.concatenate(layers, axis=1)


def wl_embedding_fractional(FC, DTI, H=2, alpha=0.5, beta=1.0):
    """
    Same propagation scheme as wl_embedding (eq. 5, Togninalli et al. 2019),
    but the transition operator is built from the *regularized fractional*
    structural graph (Filippo & Mazza 2026, Def. 4.1) instead of DTI itself:
    every real structural edge keeps its own weight, and every other node
    pair gets a power-law-decaying long-range weight on top of it. This lets
    a single WL hop already reach non-adjacent regions through the
    structural graph, rather than requiring H successive local hops — see
    fractional.regularized_fractional_weight for why the *regularized*
    variant (not the raw fractional graph) is used: it is the one Filippo &
    Mazza prove stays superdiffusive regardless of the subject's own
    algebraic connectivity.

    FC    : (N, N) — each row is the node's initial attribute vector
    DTI   : (N, N) — subject's structural (SC) weight matrix
    H     : number of WL iterations
    alpha : float in (0, 1), fractional order (0 -> most non-local, 1 -> local/DTI)
    beta  : float >= 1, non-local scaling (Filippo & Mazza Def. 4.1)
    → (N, N*(H+1)) concatenated embedding
    """
    N = FC.shape[0]
    np.fill_diagonal(DTI, 0)

    rW, _, _ = regularized_fractional_weight(DTI, alpha, beta=beta)

    deg = rW.sum(axis=1, keepdims=True)
    deg[deg == 0] = 1
    W = rW / deg

    a = FC.copy()
    layers = [a.copy()]

    for _ in range(H):
        aux = 0.5 * (a + W @ a)
        layers.append(aux.copy())
        a = aux

    return np.concatenate(layers, axis=1)
