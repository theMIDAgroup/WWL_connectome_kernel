"""Regularized fractional graph Laplacian for non-local structural propagation.

Implements the (scaled) regularized fractional graph Laplacian of Filippo &
Mazza, "Spectral and computational aspects of a regularized fractional
Laplacian for non-local diffusion on graphs", J. Numer. Math. 2026,
doi:10.1515/jnma-2026-0007 (Sec. 3.1 and Def. 4.1 / Thm. 4.7).

Used as a drop-in alternative propagation topology for the connectome WWL
embedding (embedding.wl_embedding_fractional): instead of letting the WL
step see only H-hop structural neighbours, the (regularized) fractional
Laplacian lets a single propagation step reach the whole structural graph,
with edge weight decaying by a power law in hop distance, while the
*regularized* variant (the one used everywhere here) provably keeps the
subject's real structural edges intact and is superdiffusive for small
enough alpha on essentially any graph (Filippo & Mazza, Thm. 4.12). 
"""

import numpy as np


def graph_laplacian(W):
    """Combinatorial Laplacian Delta = D - W of a symmetric weight matrix W."""
    W = np.asarray(W, dtype=np.float64)
    deg = W.sum(axis=1)
    return np.diag(deg) - W


def fractional_laplacian(Delta, alpha):
    """Spectral fractional power Delta^alpha = Q diag(lambda^alpha) Q^T.

    Delta must be symmetric PSD (the Laplacian of an undirected graph).
    alpha in (0, 1); alpha -> 1 recovers Delta, alpha -> 0 recovers the
    (graph-independent) matrix I - (1/n) 11^T (Filippo & Mazza, Lemma 3.2/3.3).

    alpha outside (0, 1] is refused, not just discouraged: for alpha in
    (0, 1) Delta^alpha is guaranteed off-diagonal <= 0 (a genuine
    non-negative-weight graph, via Bochner subordination); this guarantee
    is specific to that range and already fails at alpha=1.5 or 2 (e.g. on
    a 3-node path, the two non-adjacent nodes get a *negative* implied
    edge weight), so nothing downstream (regularized_fractional_weight,
    Prop. 3.1's power-law bound) is meaningful there.
    """
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"alpha must be in (0, 1] for Delta^alpha to remain a "
                          f"valid non-negative-weight graph Laplacian; got {alpha}.")
    vals, Q = np.linalg.eigh(Delta)
    vals = np.clip(vals, 0.0, None)  # guard tiny negative numerical noise
    return (Q * (vals ** alpha)) @ Q.T


def regularized_fractional_weight(W, alpha, beta=1.0):
    """(Scaled) regularized fractional weight matrix rW^alpha (Def. 4.1).

    Keeps the subject's own structural weight w_ij on every edge already
    present in W, and assigns beta * w^alpha_ij (the fractional-graph weight)
    to every pair with no direct structural edge, so short-range structure
    is untouched and long-range "jumps" are added on top of it, rather than
    replacing it as the raw fractional graph does.

    Parameters
    ----------
    W     : (N, N) symmetric structural weight matrix, 0 on absent edges,
            0 diagonal.
    alpha : float in (0, 1), fractional order.
    beta  : float >= 1, scaling of the non-local component (Def. 4.1); beta=1
            is the unscaled regularized fractional graph.

    Returns
    -------
    rW, Delta, Delta_alpha : the regularized weight matrix and the two
    Laplacians it was built.
    """
    W = np.asarray(W, dtype=np.float64)
    Delta = graph_laplacian(W)
    Delta_a = fractional_laplacian(Delta, alpha)

    W_a = -Delta_a.copy()
    np.fill_diagonal(W_a, 0.0)
    np.clip(W_a, 0.0, None, out=W_a)  # off-diagonal entries of Delta^alpha are <=0 by construction

    A = (W > 0).astype(np.float64)
    np.fill_diagonal(A, 0.0)

    rW = W * A + beta * W_a * (1.0 - A)
    np.fill_diagonal(rW, 0.0)
    return rW, Delta, Delta_a


def algebraic_connectivity(Delta):
    """lambda_{n-1}(Delta): smallest strictly-positive eigenvalue of a connected
    graph's Laplacian (2nd smallest overall, since lambda_n(Delta)=0)."""
    vals = np.sort(np.linalg.eigvalsh(Delta))
    return float(vals[1]) if len(vals) > 1 else 0.0
