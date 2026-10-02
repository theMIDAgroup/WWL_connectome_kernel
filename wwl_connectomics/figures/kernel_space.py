"""Kernel-matrix and (kernel-)PCA figures, each function saves exactly one PNG."""

import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch

import matplotlib.pyplot as plt

from ..style import apply_style, get_palette


def plot_kernel_matrix(K, save_path, sample_labels=None, group_sizes=None, annotate=True,
                        max_labelled_samples=40):
    """
    Heatmap of a kernel (or similarity) matrix.

    sample_labels : optional per-sample short tag shown under each tick (e.g. group).

    group_sizes   : optional OrderedDict/dict name->n (in sample order) to draw
                    dashed amber blocks + labels around the intra-group blocks.

    max_labelled_samples : above this sample count, per-sample tick labels and
                    per-cell value annotations are skipped (both become
                    unreadable clutter that hides the heatmap itself well
                    before S reaches a few hundred), group boundaries from
                    group_sizes remain the primary way to read the matrix.
    """
    P = get_palette(); apply_style()
    cmap = LinearSegmentedColormap.from_list("wwl", [P["WHITE"], P["LBLUE"], P["BLUE"], P["NAVY"]])
    S = K.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))
    # Scale off the off-diagonal values only: the diagonal is always exactly 1
    # (self-similarity), and a fixed [0, 1] range crushes all real
    # inter-subject contrast into a thin sliver when off-diagonal similarities
    # cluster tightly (as they typically do after lambda calibration).
    off_diag = K[~np.eye(S, dtype=bool)]
    vmin, vmax = (off_diag.min(), off_diag.max()) if off_diag.size else (0, 1)
    im = ax.imshow(K, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, label="kernel similarity", shrink=0.85)
    ax.set_title("Kernel matrix", fontsize=11, color=P["NAVY"], pad=10)

    if S <= max_labelled_samples:
        ticks = list(range(S))
        if sample_labels is not None:
            labels = [f"S{i+1}\n({sample_labels[i]})" for i in ticks]
        else:
            labels = [f"S{i+1}" for i in ticks]
        ax.set_xticks(ticks); ax.set_xticklabels(labels, fontsize=7.5, rotation=45)
        ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.5)

        if annotate:
            mid = (vmin + vmax) / 2
            for i in range(S):
                for j in range(S):
                    ax.text(j, i, f"{K[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color=P["WHITE"] if K[i, j] > mid else P["NAVY"])
    else:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(f"{S} subjects", color=P["GRAY"], fontsize=9)
        ax.set_ylabel(f"{S} subjects", color=P["GRAY"], fontsize=9)

    if group_sizes:
        bounds = np.cumsum([0] + list(group_sizes.values()))
        for name, x0, size in zip(group_sizes.keys(), bounds[:-1], group_sizes.values()):
            rect = plt.Rectangle((x0 - .5, x0 - .5), size, size,
                                  linewidth=2, edgecolor=P["AMBER"],
                                  facecolor="none", linestyle="--")
            ax.add_patch(rect)
            ax.text(x0 + size / 2 - 0.5, -1.2, str(name), ha="center",
                    fontsize=8, color=P["AMBER"], fontweight="bold")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_kernel_pca(Z, group_labels, save_path, group_colors=None, sample_names=None):
    """2D (kernel-)PCA scatter of samples, colored by a categorical group label."""
    P = get_palette(); apply_style()
    S = len(group_labels)
    cats = sorted(set(group_labels), key=str)
    if group_colors is None:
        if len(cats) <= 2:
            palette_cycle = [P["BLUE"], P["CORAL"]]
        else:
            cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
            palette_cycle = [cmap(i % cmap.N) for i in range(len(cats))]
        group_colors = dict(zip(cats, palette_cycle))

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title("Kernel PCA embedding space\n(color = group)",
                  fontsize=11, color=P["NAVY"], pad=8)
    for i in range(S):
        ax.scatter(Z[i, 0], Z[i, 1], c=[group_colors[group_labels[i]]], s=200,
                   edgecolors=P["NAVY"], linewidths=1.2, zorder=3)
        label = sample_names[i] if sample_names is not None else f"S{i+1}"
        ax.annotate(label, xy=Z[i], xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=P["NAVY"])
    ax.legend(handles=[Patch(facecolor=group_colors[c], label=str(c)) for c in cats],
              fontsize=9, framealpha=0.9, edgecolor=P["LGRAY"], loc="best")
    ax.set_xlabel("kPC1", color=P["GRAY"])
    ax.set_ylabel("kPC2", color=P["GRAY"])
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axhline(0, color=P["LGRAY"], linewidth=0.8)
    ax.axvline(0, color=P["LGRAY"], linewidth=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_kernel_pca_colored(Z, values, col_name, save_path):
    """
    2D (kernel-)PCA scatter colored by an arbitrary variable, missing values shown in light gray.
    """
    apply_style()
    is_missing = pd.isna(values) if not isinstance(values[0], str) else pd.Series(values).isna().values
    values_arr = pd.Series(values)
    n_unique = values_arr.dropna().nunique()
    is_numeric = pd.api.types.is_numeric_dtype(values_arr) and n_unique > 6

    fig, ax = plt.subplots(figsize=(7, 6))

    if is_numeric:
        mask = ~values_arr.isna().values
        sc = ax.scatter(Z[mask, 0], Z[mask, 1], c=values_arr[mask].astype(float),
                         cmap="viridis", s=45, alpha=0.85,
                         edgecolors="white", linewidths=0.4)
        plt.colorbar(sc, ax=ax, label=col_name)
        if (~mask).sum():
            ax.scatter(Z[~mask, 0], Z[~mask, 1], c="lightgray", s=35,
                       alpha=0.6, label=f"missing (n={(~mask).sum()})")
            ax.legend(fontsize=8)
    else:
        cats = sorted(values_arr.dropna().unique().tolist(), key=str)
        cmap = plt.get_cmap("tab10" if len(cats) <= 10 else "tab20")
        for i, cat in enumerate(cats):
            mask = (values_arr == cat).values
            ax.scatter(Z[mask, 0], Z[mask, 1], color=cmap(i % cmap.N),
                       s=45, alpha=0.85, edgecolors="white", linewidths=0.4,
                       label=f"{cat} (n={mask.sum()})")
        mask_na = values_arr.isna().values
        if mask_na.sum():
            ax.scatter(Z[mask_na, 0], Z[mask_na, 1], c="lightgray", s=35,
                       alpha=0.6, label=f"missing (n={mask_na.sum()})")
        ax.legend(fontsize=8, framealpha=0.85, loc="best")

    ax.set_xlabel("PC1 (kernel PCA)")
    ax.set_ylabel("PC2 (kernel PCA)")
    ax.set_title(f"Kernel PCA colored by '{col_name}'", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_lambda_sensitivity(lam_grid, accs, save_path, lam_mu=None, lam_fisher=None, lam_cv_best=None):
    """Balanced accuracy vs lambda (log scale), with optional reference lambda markers."""
    P = get_palette(); apply_style()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.semilogx(lam_grid, accs, color=P["TEAL"], lw=2, zorder=3)
    for lam, color, ls, label in [
        (lam_mu,      P["AMBER"], "--", "1/μ"),
        (lam_fisher,  P["CORAL"], ":",  "Fisher"),
        (lam_cv_best, P["NAVY"],  "-",  "CV-best"),
    ]:
        if lam is not None:
            ax.axvline(lam, color=color, lw=1.8, linestyle=ls,
                       label=f"{label} (λ={lam:.3f})", zorder=4)
    ax.axhline(0.5, color=P["LGRAY"], lw=1, linestyle=":")
    ax.fill_between(lam_grid, accs, 0.5, where=accs > 0.5, alpha=0.12, color=P["TEAL"])
    ax.set_xlabel("λ (log scale)", fontsize=9)
    ax.set_ylabel("Bal. accuracy (CV)", fontsize=9)
    ax.set_title("λ sensitivity", fontsize=10, color=P["NAVY"])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.22)
    ax.set_ylim(0.35, 1.05)
    ax.set_xlim(lam_grid[0], lam_grid[-1])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path
