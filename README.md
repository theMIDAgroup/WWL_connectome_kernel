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
- **Distance**: `wwl_connectomics.distances.build_D` — pairwise Wasserstein-1
  distance between embeddings (parallelized).
- **Kernel**: `wwl_connectomics.kernels` — Laplacian kernel `K = exp(-λD)`,
  with `1/μ`, Fisher-separability, or nested-CV bandwidth calibration.
- **Cross-validation**: `wwl_connectomics.crossval` — nested stratified CV
  with joint `(λ, C)` grid search, WL-depth (`H`) selection, permutation
  testing, and the graph-theory / shortest-path / WL-subtree baselines.
- **Figures**: `wwl_connectomics.figures` — every plotting function saves
  exactly one atomic PNG; `wwl_connectomics.reports` composes them into the
  combined multi-panel layouts.

Reference: Togninalli, M., Ghisu, E., Llinares-López, F., Rieck, B.,
Borgwardt, K. *Wasserstein Weisfeiler-Lehman Graph Kernels*. NeurIPS 2019.

## License

Not yet chosen — add a `LICENSE` file before making this repository public.
