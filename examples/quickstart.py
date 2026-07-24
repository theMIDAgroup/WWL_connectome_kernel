"""
quickstart.py
==============
Minimal end-to-end usage of the WWL connectome kernel: given a collection of
per-subject structural (SC) and functional (FC) connectivity matrices plus a
group label, build the kernel and classify.

This script assumes you already have your connectomes as three NumPy arrays:

    SC : (S, N, N)  structural connectivity  (e.g. streamline counts / FA)
    FC : (S, N, N)  functional connectivity  (e.g. BOLD correlation)
    y  : (S,)       integer group label per subject

The only thing specific to this script is HOW those three arrays are
obtained: below, `load_your_connectomes()` fabricates a small toy dataset
with `wwl_connectomics.synthetic` purely so this file runs standalone with
no external data. Replace that one function with your own loading code
(e.g. `np.load(...)` on your own SC/FC files) — everything after it is the
actual library usage and does not change.

Run:
    python examples/quickstart.py
"""

import os

import numpy as np

from wwl_connectomics.crossval import nested_cv_kernel
from wwl_connectomics.distances import build_D, kernel_pca_2d
from wwl_connectomics.embedding import wl_embedding
from wwl_connectomics.figures import plot_kernel_pca
from wwl_connectomics.kernels import build_K, calibrate_lam
from wwl_connectomics.synthetic import make_group_A, make_group_B


def load_your_connectomes(n_per_group=15, n_regions=30):
    """
    Replace this function with your own data loading, e.g.:

        SC = np.load("my_SC.npy")   # (S, N, N)
        FC = np.load("my_FC.npy")   # (S, N, N)
        y  = np.load("my_labels.npy")  # (S,) int, e.g. 0=CN, 1=AD

    Here we fabricate a small two-group toy set instead, only so the script
    is runnable with no data of your own.
    """
    FC_list, SC_list = [], []
    for i in range(n_per_group):
        fc, sc = make_group_A(seed=i, n=n_regions)
        FC_list.append(fc); SC_list.append(sc)
    for i in range(n_per_group):
        fc, sc = make_group_B(seed=i, n=n_regions)
        FC_list.append(fc); SC_list.append(sc)

    SC = np.stack(SC_list)
    FC = np.stack(FC_list)
    y = np.array([0] * n_per_group + [1] * n_per_group)
    return SC, FC, y


def main():
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "quickstart_out")
    os.makedirs(out_dir, exist_ok=True)

    SC, FC, y = load_your_connectomes()
    S, N, _ = SC.shape
    print(f"{S} subjects, {N} regions, {len(np.unique(y))} groups")

    # 1. WL embedding: DTI/SC drives propagation, FC is the propagated signal.
    print("Computing WL embeddings (H=2)...")
    H = 2
    embeddings = [wl_embedding(FC[i], SC[i].copy(), H) for i in range(S)]

    # 2. Pairwise Wasserstein-1 distance between embeddings.
    print("Building the Wasserstein distance matrix...")
    D = build_D(embeddings, n_jobs=-1)

    # 3. Nested cross-validated classification (lambda and C jointly tuned
    #    on each training fold — no test-set information leaks into the
    #    kernel).
    print("Nested CV classification...")
    mean_bacc, std_bacc = nested_cv_kernel(D, y, n_outer=5, n_inner=3, seed=42, lam_method="cv")
    print(f"Balanced accuracy: {mean_bacc:.3f} +/- {std_bacc:.3f}")

    # 4. Kernel PCA visualization (unsupervised bandwidth, for a quick look
    #    at the embedding space — not the same lambda used for classification).
    lam = calibrate_lam(D, y_train=None, method="1/mu")
    K = build_K(D, lam)
    Z = kernel_pca_2d(K)
    fig_path = os.path.join(out_dir, "kernel_pca.png")
    plot_kernel_pca(Z, list(y), fig_path)
    print(f"Kernel PCA figure saved to {fig_path}")


if __name__ == "__main__":
    main()
