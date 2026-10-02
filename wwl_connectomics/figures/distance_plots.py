"""Distance-matrix and intra/inter-group distance distribution figures.
"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu

import matplotlib.pyplot as plt

from ..style import apply_style, get_palette


def _default_colors(P, names):
    cycle = [P["BLUE"], P["CORAL"], P["GRAY"], P["TEAL"], P["AMBER"], P["VIOLET"]]
    return {n: cycle[i % len(cycle)] for i, n in enumerate(names)}


def plot_distance_heatmap(D, save_path, group_sizes=None, title="Distance matrix"):
    """Heatmap of a (S, S) distance matrix, with optional group divider lines."""
    P = get_palette(); apply_style()
    cmap = LinearSegmentedColormap.from_list("dist", [P["WHITE"], P["LBLUE"], P["NAVY"]])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    # Scale off the off-diagonal values only: the diagonal is always exactly 0
    # (self-distance), and including it in the color range crushes all real
    # inter-subject contrast into a thin sliver when off-diagonal distances
    # cluster tightly far away from 0 (as they typically do).
    off_diag = D[~np.eye(len(D), dtype=bool)]
    vmin, vmax = (off_diag.min(), off_diag.max()) if off_diag.size else (None, None)
    im = ax.imshow(D, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, shrink=0.85, label="distance")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Subject"); ax.set_ylabel("Subject")

    if group_sizes:
        bounds = np.cumsum(list(group_sizes.values()))
        for b in bounds[:-1]:
            ax.axhline(b - 0.5, color=P["CORAL"], lw=1.5)
            ax.axvline(b - 0.5, color=P["CORAL"], lw=1.5)
        start = 0
        for name, n in group_sizes.items():
            ax.text(start + n / 2, -len(D) * 0.03, str(name), ha="center", fontsize=9,
                    color=P["NAVY"], fontweight="bold")
            start += n

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_distance_boxplot_by_group(distance_groups, save_path, colors=None, ylabel="distance"):
    """Boxplot of a distance distribution per named group."""
    P = get_palette(); apply_style()
    colors = colors or _default_colors(P, distance_groups.keys())
    fig, ax = plt.subplots(figsize=(5.5, 5))
    names = list(distance_groups.keys())
    bp = ax.boxplot([distance_groups[n] for n in names], patch_artist=True, labels=names)
    for patch, n in zip(bp["boxes"], names):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.6)
    ax.set_ylabel(ylabel)
    ax.set_title("Distance distributions by group", fontsize=10)
    ax.grid(True, alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_distance_intra_inter_hist(distance_groups, save_path, colors=None,
                                    xlabel="$W_1$ distance", test_pair=None, alternative="less"):
    """
    Overlaid density histograms, one per named group, dashed mean lines.
    test_pair=(name_a, name_b): runs a one-sided Mann-Whitney and shows it in the title.
    """
    P = get_palette(); apply_style()
    colors = colors or _default_colors(P, distance_groups.keys())
    fig, ax = plt.subplots(figsize=(6, 5))
    all_vals = [v for vals in distance_groups.values() for v in vals]
    bins = np.linspace(min(all_vals), max(all_vals), 32)
    for name, vals in distance_groups.items():
        ax.hist(vals, bins=bins, alpha=0.55, color=colors[name], density=True,
                label=f"{name}  μ={np.mean(vals):.2f}")
        ax.axvline(np.mean(vals), color=colors[name], lw=1.8, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.85)

    title = "Distance distributions"
    if test_pair is not None:
        a, b = test_pair
        _, p_mw = mannwhitneyu(distance_groups[a], distance_groups[b], alternative=alternative)
        stars = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."
        title += f"  |  {a} {alternative} {b}: {stars} (p={p_mw:.3f})"
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_distance_stats_table(distance_groups, save_path, mw_pair=None, alternative="less"):
    """Text table: mean/std/median/IQR per group, plus an optional Mann-Whitney row."""
    P = get_palette(); apply_style()
    fig, ax = plt.subplots(figsize=(6, 0.9 + 0.5 * (len(distance_groups) + 2)))
    ax.axis("off")

    rows = [["", "Mean", "Std", "Median", "IQR"]]
    for name, vals in distance_groups.items():
        vals = np.asarray(vals)
        rows.append([str(name), f"{vals.mean():.3f}", f"{vals.std():.3f}",
                     f"{np.median(vals):.3f}",
                     f"{np.percentile(vals, 75) - np.percentile(vals, 25):.3f}"])

    if mw_pair is not None:
        a, b = mw_pair
        _, p_mw = mannwhitneyu(distance_groups[a], distance_groups[b], alternative=alternative)
        stars = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."
        rows.append(["", "", "", "", ""])
        rows.append([f"Mann-Whitney ({a} {alternative} {b})", f"p={p_mw:.4f}", stars, "", ""])

    col_widths = [0.30, 0.16, 0.16, 0.16, 0.16]
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            x_pos = sum(col_widths[:ci]) + 0.02
            weight = "bold" if ri == 0 or ci == 0 else "normal"
            ax.text(x_pos, 1 - ri * 0.14, cell, transform=ax.transAxes,
                    fontsize=9.5, fontweight=weight, color="black", va="top")
    ax.set_title("Distance statistics", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path
