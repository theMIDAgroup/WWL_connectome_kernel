# wwl-connectome-kernel

A Wasserstein Weisfeiler–Lehman (WWL) graph kernel framework for joint
structural (DTI) + functional (fMRI) brain connectome analysis.

Each subject is represented as a single attributed graph: DTI tractography
defines the propagation topology, and the functional connectivity profile is
the signal propagated over it (Weisfeiler–Lehman continuous propagation,
Togninalli et al. 2019). Subject-subject similarity is the Laplacian kernel
of the Wasserstein-1 distance between the resulting multi-scale node
embeddings, usable directly by any kernel-based classifier and for
visualization (kernel PCA/LDA, embedding shift, optimal transport plans).

Propagation can use either the direct structural graph (`H`-hop local
averaging) or a non-local alternative, the regularized fractional graph
Laplacian of Filippo & Mazza 2026: every real structural edge keeps its own
weight, and every other pair of regions gets an additional, power-law-decaying
long-range weight, so a single WL step already reaches the whole structural
graph instead of only its `H`-hop neighbourhood. The two are combined the
same way everywhere in the package — `wl_embedding` vs.
`wl_embedding_fractional`, `select_best_h` vs. `select_best_h_alpha` — so
switching between them never changes any other part of the pipeline.

## Repository layout

```
wwl_connectomics/     the package — core, reusable code (embedding, distances,
                       kernels, cross-validation, atomic plotting functions)
examples/               a single minimal, runnable usage example
pyproject.toml
```

No data or data-generation code is included. The package is meant to be
applied to a connectome collection you already have.

## Installation

```bash
pip install -e .
```

Installs `wwl_connectomics` in editable mode plus its dependencies (numpy,
scipy, pandas, scikit-learn, networkx, matplotlib, nilearn, joblib). Python
≥3.9.

## Quickstart

```bash
python examples/quickstart.py
```

[`examples/quickstart.py`](examples/quickstart.py) shows the full pipeline —
WL embedding → Wasserstein distance matrix → kernel → nested-CV
classification → kernel-PCA figure — starting from three plain NumPy arrays:

```python
SC : (S, N, N)  structural connectivity, one matrix per subject
FC : (S, N, N)  functional connectivity, one matrix per subject
y  : (S,)       integer group label per subject
```

The only part specific to the example is `load_your_connectomes()`, which
fabricates a tiny toy dataset so the script runs with no data of your own.
Replace that one function with however you already load your SC/FC matrices
(e.g. `np.load(...)`) — everything after it is the actual library usage and
does not change.

## Method summary

- **Embedding**: `wwl_connectomics.embedding.wl_embedding` — WL continuous
  propagation of FC node attributes along DTI-weighted structural edges,
  `H` iterations, stacked into one `(N, N·(H+1))` point cloud per subject.
  `wl_embedding_fractional` is the drop-in non-local variant (same
  signature plus `alpha`, `beta`), built on `wwl_connectomics.fractional`.
- **Fractional propagation**: `wwl_connectomics.fractional` — regularized
  fractional graph Laplacian (`fractional_laplacian`,
  `regularized_fractional_weight`); `alpha in (0, 1]` interpolates between
  the local operator (`alpha -> 1`) and an entirely non-local,
  topology-blind one (`alpha -> 0`), `beta >= 1` scales the non-local
  component. Guaranteed superdiffusive (never slower-mixing than plain DTI
  propagation) for small enough `alpha` on essentially any graph — the
  unregularized fractional Laplacian does not have that guarantee in
  general (see the module docstring for the precise condition).
- **Distance**: `wwl_connectomics.distances.build_D` — pairwise Wasserstein-1
  distance between embeddings (parallelized).
- **Kernel**: `wwl_connectomics.kernels` — Laplacian kernel `K = exp(-λD)`,
  with `1/μ`, Fisher-separability, or nested-CV bandwidth calibration; the
  joint `(λ, C)` grid search is parallelized.
- **Cross-validation**: `wwl_connectomics.crossval` — nested stratified CV
  with joint `(λ, C)` grid search, WL-depth (`H`) selection (`select_best_h`)
  or joint `(H, alpha)` selection for fractional propagation
  (`select_best_h_alpha`), permutation testing, and the graph-theory /
  shortest-path / WL-subtree baselines.
- **Figures**: `wwl_connectomics.figures` — every plotting function saves
  exactly one atomic PNG; `wwl_connectomics.reports` composes them into the
  combined multi-panel layouts.

References: Togninalli, M., Ghisu, E., Llinares-López, F., Rieck, B.,
Borgwardt, K. *Wasserstein Weisfeiler-Lehman Graph Kernels*. NeurIPS 2019.
Filippo, A., Mazza, M. *Spectral and computational aspects of a regularized
fractional Laplacian for non-local diffusion on graphs*. J. Numer. Math.
2026, doi:10.1515/jnma-2026-0007.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode.en) — see [`LICENSE`](LICENSE).

