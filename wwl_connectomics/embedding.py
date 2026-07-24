"""Weisfeiler-Lehman continuous embedding propagation (eq. 5, Togninalli et al. 2019)."""

import numpy as np


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
