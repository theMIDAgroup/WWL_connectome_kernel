"""Regression-result figures (continuous target, e.g. MMSE) — one PNG per function."""

import numpy as np
from matplotlib.patches import Patch
from scipy.stats import linregress

import matplotlib.pyplot as plt

from ..style import apply_style, get_palette


def plot_regression_scatter(y_true, y_pred, save_path, label="target", r2=None, pearson_r=None):
    """True vs cross-validated predicted value, with the y=x reference line."""
    P = get_palette(); apply_style()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(y_true, y_pred, s=45, alpha=0.75, color=P["BLUE"],
               edgecolors=P["NAVY"], linewidths=0.4, zorder=3)
    lo = min(np.min(y_true), np.min(y_pred))
    hi = max(np.max(y_true), np.max(y_pred))
    ax.plot([lo, hi], [lo, hi], color=P["GRAY"], lw=1.3, linestyle="--", zorder=2,
            label="y = x (perfect prediction)")
    subtitle = []
    if r2 is not None:
        subtitle.append(f"R²={r2:.3f}")
    if pearson_r is not None:
        subtitle.append(f"r={pearson_r:.3f}")
    title = f"{label}: predicted (CV) vs true"
    if subtitle:
        title += "  (" + ", ".join(subtitle) + ")"
    ax.set_title(title, fontsize=10.5, color=P["NAVY"], fontweight="bold")
    ax.set_xlabel(f"{label} (true)", color=P["GRAY"])
    ax.set_ylabel(f"{label} (predicted, out-of-fold)", color=P["GRAY"])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_scatter_regression(x, y, save_path, xlabel="x", ylabel="y", title=None):
    """
    Scatter tra due variabili REALI (non predetta vs vera) con retta di
    regressione OLS (scipy.stats.linregress) e r/p annotati. Uso tipico:
    x = shift di embedding (di una regione o aggregato), y = variabile
    clinica reale (es. MMSE) — per vedere la correlazione grezza, non una
    performance di modello.
    """
    P = get_palette(); apply_style()
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    res = linregress(x, y)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(x, y, s=40, alpha=0.65, color=P["BLUE"],
               edgecolors=P["NAVY"], linewidths=0.3, zorder=3)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, res.intercept + res.slope * xs, color=P["CORAL"], lw=2, zorder=4,
            label=f"OLS: r={res.rvalue:.3f}, p={res.pvalue:.2e}")
    ax.set_xlabel(xlabel, color=P["GRAY"])
    ax.set_ylabel(ylabel, color=P["GRAY"])
    ax.set_title(title or f"{ylabel} vs {xlabel}", fontsize=10.5, color=P["NAVY"], fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_regions_scatter_grid(shift, y, region_names, save_path, top_n=12,
                               target_label="MMSE", n_cols=4, ranked_idx=None):
    """
    Griglia (piccoli multipli) di scatter shift-regione vs target, uno per
    regione, con retta OLS e r/p per pannello — un solo PNG.

    ranked_idx : indici delle regioni da mostrare, in ordine (es. per |r|
                 decrescente); se None, calcolate qui da zero via |pearson r|.
    """
    P = get_palette(); apply_style()
    shift = np.asarray(shift, dtype=float); y = np.asarray(y, dtype=float)
    N = shift.shape[1]

    if ranked_idx is None:
        rs = np.array([linregress(shift[:, r], y).rvalue for r in range(N)])
        ranked_idx = np.argsort(np.abs(rs))[::-1]
    top_idx = list(ranked_idx[:min(top_n, N)])

    n_rows = int(np.ceil(len(top_idx) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, top_idx):
        xr = shift[:, r]
        res = linregress(xr, y)
        ax.scatter(xr, y, s=18, alpha=0.55, color=P["BLUE"],
                   edgecolors="none", zorder=3)
        xs = np.linspace(xr.min(), xr.max(), 50)
        ax.plot(xs, res.intercept + res.slope * xs, color=P["CORAL"], lw=1.6, zorder=4)
        name = str(region_names[r]).replace("7Networks_", "").replace("_", " ")
        ax.set_title(f"{name}\nr={res.rvalue:.2f}, p={res.pvalue:.1e}", fontsize=8.5, color=P["NAVY"])
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    for ax in axes[len(top_idx):]:
        ax.axis("off")

    fig.suptitle(f"Top {len(top_idx)} regions by |r|, shift vs {target_label}",
                 fontsize=12, color=P["NAVY"], fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_region_importance_bar(coefs, region_names, save_path, top_n=20,
                                title="Top regions by |coefficient|", xlabel="Coefficient"):
    """Top-N regions by |coefs| (e.g. standardized linear-SVR weights), horizontal bars."""
    P = get_palette(); apply_style()
    coefs = np.asarray(coefs)
    top_n = min(top_n, len(coefs))
    top_idx = np.argsort(np.abs(coefs))[::-1][:top_n]
    top_vals = coefs[top_idx]
    top_names = [str(region_names[i]).replace("7Networks_", "").replace("_", " ") for i in top_idx]
    colors_bar = [P["CORAL"] if v > 0 else P["BLUE"] for v in top_vals]
    y_pos = np.arange(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.32 * top_n + 1.5))
    ax.barh(y_pos, top_vals, color=colors_bar, alpha=0.82, edgecolor=P["WHITE"], linewidth=0.5)
    ax.axvline(0, color=P["NAVY"], lw=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel(xlabel, color=P["GRAY"])
    ax.set_title(title, fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=P["CORAL"], label="Higher shift -> higher target"),
        Patch(facecolor=P["BLUE"],  label="Higher shift -> lower target"),
    ], fontsize=7.5, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path
