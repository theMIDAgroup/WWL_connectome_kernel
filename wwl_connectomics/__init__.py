"""
wwl_connectomics
=================
Wasserstein Weisfeiler-Lehman (WWL) kernel pipeline for comparing structural
(DTI) / functional (fMRI) connectomes: node embedding propagation, W1/W2
distances, kernel construction + nested-CV classification, permutation
testing, and a library of atomic (one-figure-per-function) plots.

Submodules
----------
style       - color palette and matplotlib rcParams
atlas       - Schaefer-100 / 7-Yeo-network layout (single source of truth)
synthetic   - synthetic FC/DTI group generators (demos, smoke tests)
io          - matrix loading/normalization, anagrafica loading, scenario file discovery
embedding   - wl_embedding (WL propagation), wl_embedding_fractional (regularized
              fractional-Laplacian propagation, Filippo & Mazza 2026)
fractional  - regularized fractional graph Laplacian (Filippo & Mazza 2026)
distances   - wasserstein_1/2, build_D, kernel_pca_2d
kernels     - calibrate_lam, build_K, METHOD_ORDER/METHOD_LABELS
crossval    - nested_cv_kernel/_flat/_subtree, permutation_test, select_best_h,
              select_best_h_alpha (fractional-propagation sweep)
labels      - real-data session selection + anagrafica target-label extraction
shifts      - per-region embedding shift + group statistics
figures.*   - one function per standalone, saveable figure
reports     - convenience wrappers combining several atomic figures into the
              classic multi-panel layouts (kept for backward compatibility)
"""

from . import (
    atlas, crossval, distances, embedding, figures, fractional, io, kernels,
    labels, reports, shifts, style, synthetic,
)

__all__ = [
    "atlas", "crossval", "distances", "embedding", "figures", "fractional",
    "io", "kernels", "labels", "reports", "shifts", "style", "synthetic",
]
