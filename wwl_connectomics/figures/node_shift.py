"""
Per-region embedding shift figures — the biggest source of duplication in the
original codebase (wwl_benchmark.plot_population_shift, make_scenario_figures'
copy of the same panels, and wwl_real_node_shift.plot_node_shift_ranking all
reimplemented near-identical top-20/boxplot/scatter/significant-fraction
panels). Each panel is now one atomic function operating on plain arrays, so
a single canonical implementation is shared everywhere.

Two families of functions here, matching the two ways "network" is
represented in this codebase:
  - subject-pair functions (plot_shift_comparison_subject,
    plot_shift_distribution_by_network, plot_mean_shift_heatmap) take the
    `networks` dict-of-index-ranges format (atlas.NETWORKS) and compare two
    *individual* subjects.
  - group-level functions (plot_top_regions_shift, plot_shift_boxplot_by_network,
    plot_shift_scatter_groups, plot_significant_fraction_by_network) take a
    parallel `rsn_ids` array (as found in atlas CSV files) and compare two
    *groups* of subjects (shifts_0/shifts_1 are (S, N) arrays).
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

from .. import atlas as atlas_mod
from ..style import apply_style, draw_violin, get_palette


def _project_pca(emb, N):
    """Project (iter-0, iter-H) node attributes of one subject into a shared 2D PCA space."""
    X0, XH = emb[:, :N], emb[:, -N:]
    sc = StandardScaler().fit_transform(np.vstack([X0, XH]))
    pca = PCA(n_components=2)
    X2d = pca.fit_transform(sc)
    return X2d[:N], X2d[N:], pca


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ── single-subject ───────────────────────────────────────────────────────────
def plot_embedding_shift_2d(emb, region_names, save_path, subject_label=None, h_iter=None):
    """2D PCA scatter of one subject's node attributes at iter 0 vs iter H, arrows = shift."""
    P = get_palette(); apply_style()
    N = len(region_names)
    col0, colH = P["BLUE"], P["TEAL"]

    X0_2d, XH_2d, pca = _project_pca(emb, N)
    X0, XH = emb[:, :N], emb[:, -N:]
    shift_l2 = np.linalg.norm(XH - X0, axis=1)
    shift_abs = np.mean(np.abs(XH - X0), axis=1)
    shift_norm = (shift_l2 - shift_l2.min()) / (np.ptp(shift_l2) + 1e-9)
    sizes_H = 120 + 400 * shift_norm
    order = np.argsort(shift_l2)[::-1]

    fig, ax = plt.subplots(figsize=(8, 6.2))
    title = "Node embedding shift"
    if subject_label: title += f" — {subject_label}"
    if h_iter is not None: title += f"  (iter 0 → iter {h_iter})"
    fig.suptitle(title, fontsize=12, fontweight="bold", color=P["NAVY"])

    ax.scatter(X0_2d[:, 0], X0_2d[:, 1], c=col0, s=200, marker="o",
               edgecolors=P["NAVY"], linewidths=1.2, label="iter 0 (raw FC)", zorder=3, alpha=0.9)
    h_label = f"iter {h_iter}" if h_iter is not None else "iter H"
    ax.scatter(XH_2d[:, 0], XH_2d[:, 1], c=colH, s=sizes_H, marker="s",
               edgecolors=P["NAVY"], linewidths=1.2,
               label=f"{h_label} (after propagation)\nsize ∝ shift magnitude", zorder=3, alpha=0.9)

    for i in range(N):
        ax.annotate("", xy=XH_2d[i], xytext=X0_2d[i],
                    arrowprops=dict(arrowstyle="->", color=P["GRAY"], lw=0.9, alpha=0.5,
                                     connectionstyle="arc3,rad=0.18"))
        ax.annotate(region_names[i], xy=X0_2d[i], xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=P["NAVY"], alpha=0.8)

    for i in order[:2]:
        ax.annotate(f"Δ={shift_l2[i]:.3f}", xy=XH_2d[i], xytext=(8, -14),
                    textcoords="offset points", fontsize=7.5, color=P["AMBER"],
                    arrowprops=dict(arrowstyle="-", color=P["AMBER"], lw=0.6))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", color=P["GRAY"])
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", color=P["GRAY"])
    ax.legend(fontsize=8.5, framealpha=0.9, edgecolor=P["LGRAY"], loc="best")
    ax.grid(True, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    summary = "  ".join(f"{region_names[i]}: {shift_l2[i]:.3f}" for i in order)
    ax.text(0.01, -0.06, f"L2 shift: {summary}", transform=ax.transAxes,
            fontsize=7, color=P["GRAY"], va="top")

    fig.tight_layout(pad=2.0, rect=[0, 0.04, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)

    shift_df = pd.DataFrame({
        "region": [region_names[i] for i in order],
        "shift_l2": shift_l2[order], "shift_abs": shift_abs[order],
        "rank": np.arange(1, N + 1),
    }).set_index("rank")
    return {"region": [region_names[i] for i in order], "shift_l2": shift_l2[order],
            "shift_abs": shift_abs[order], "shift_df": shift_df, "save_path": save_path}


def plot_shift_comparison_subject(emb, shift, region_names, networks, net_colors, save_path,
                                   subject_label="", h_iter=None, color=None, top_n=5):
    """One subject's node-embedding-shift PCA panel, colored by network (circle=iter0, square=iterH)."""
    P = get_palette(); apply_style()
    N = len(region_names)
    color = color or P["BLUE"]

    X0, XH, pca = _project_pca(emb, N)
    top_idx = np.argsort(shift)[::-1][:top_n]
    sn = (shift - shift.min()) / (np.ptp(shift) + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 6))
    title = "Node embedding shift"
    if h_iter is not None: title += f"  (iter 0 → iter {h_iter})"
    fig.suptitle(title, fontsize=10, color=P["NAVY"])

    ax.scatter(X0[:, 0], X0[:, 1], c=net_colors, s=40, marker="o",
               edgecolors="white", linewidths=0.5, alpha=0.5, zorder=2)
    ax.scatter(XH[:, 0], XH[:, 1], c=net_colors, s=30 + 200 * sn, marker="s",
               edgecolors=P["NAVY"], linewidths=0.5, alpha=0.85, zorder=3)
    for i in range(N):
        ax.annotate("", xy=XH[i], xytext=X0[i],
                    arrowprops=dict(arrowstyle="->", color=P["GRAY"], lw=0.5, alpha=0.3,
                                     connectionstyle="arc3,rad=0.1"))
    for i in top_idx:
        ax.annotate(region_names[i], xy=XH[i], xytext=(5, 5), textcoords="offset points",
                    fontsize=7, color=P["AMBER"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.7,
                              edgecolor=P["AMBER"], linewidth=0.5))
        ax.annotate(f"Δ={shift[i]:.2f}", xy=XH[i], xytext=(5, -12),
                    textcoords="offset points", fontsize=6.5, color=P["AMBER"])

    label = f"Subject {subject_label}" if subject_label else "Subject"
    ax.set_title(f"{label}  mean δ={shift.mean():.3f}  max δ={shift.max():.3f}",
                 fontsize=10, color=color)
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", color=P["GRAY"])
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", color=P["GRAY"])
    ax.grid(True, alpha=0.3); ax.spines[["top", "right"]].set_visible(False)

    iter_legend = ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=P["GRAY"], markersize=7, label='iter 0'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=P["GRAY"], markersize=7,
               label=f'iter {h_iter}' if h_iter is not None else 'iter H'),
    ], loc="lower left", fontsize=8, framealpha=0.85, edgecolor=P["LGRAY"])
    ax.add_artist(iter_legend)
    ax.legend(handles=atlas_mod.legend_handles(networks), loc="upper right", fontsize=7,
              framealpha=0.85, ncol=1, edgecolor=P["LGRAY"], title="Network", title_fontsize=7.5)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    print(f"  {save_path}")
    return save_path


def plot_shift_distribution_by_network(shift_A, shift_B, networks, save_path, label_A="A", label_B="B"):
    """Beeswarm of per-region shift for two individual subjects, grouped by network, mean±std markers."""
    P = get_palette(); apply_style()
    net_names = list(networks.keys())
    nn = len(net_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle("Embedding shift  $\\delta(r_i)=\\|a^H(r_i)-a^0(r_i)\\|_2$", fontsize=11, color=P["NAVY"])

    rng_j = np.random.RandomState(0)
    for ni, (net, (idxs, col)) in enumerate(networks.items()):
        idxs = list(idxs)
        for sh, mk, fc, off in [(shift_A, "o", P["BLUE"], 0.30), (shift_B, "s", P["CORAL"], -0.30)]:
            vals = sh[idxs]
            jit = rng_j.uniform(-0.22, 0.22, len(vals))
            ax.scatter(vals, ni + jit, color=fc, marker=mk, s=28, alpha=0.75,
                       edgecolors="white", linewidths=0.4, zorder=3)
            mn, sd = vals.mean(), vals.std()
            ax.errorbar(mn, ni + off, xerr=sd, fmt="none", color=fc, lw=1.5, capsize=3, zorder=4)
            ax.plot(mn, ni + off, marker="|", color=fc, markersize=10, markeredgewidth=2, zorder=5)
        ax.axhspan(ni - 0.48, ni + 0.48, color=col, alpha=0.07, zorder=0)

    ax.set_yticks(range(nn))
    ax.set_yticklabels([f"{n}  ({len(list(networks[n][0]))} ROIs)" for n in net_names], fontsize=9)
    ax.set_xlabel("Embedding shift  $\\delta(r_i)$", color=P["GRAY"])
    ax.set_title("Distribution per network", fontsize=10, color=P["NAVY"])
    ax.grid(axis="x", alpha=0.35); ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    ax.legend(handles=[
        Line2D([0], [0], marker='o', color='w', markerfacecolor=P["BLUE"], markersize=8, label=f"Subject {label_A}"),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=P["CORAL"], markersize=8, label=f"Subject {label_B}"),
        Line2D([0], [0], marker='|', color=P["BLUE"], markersize=10, markeredgewidth=2, label=f"mean±std {label_A}"),
        Line2D([0], [0], marker='|', color=P["CORAL"], markersize=10, markeredgewidth=2, label=f"mean±std {label_B}"),
    ], loc="lower center", fontsize=8, framealpha=0.9, edgecolor=P["LGRAY"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    print(f"  {save_path}")
    return save_path


def plot_mean_shift_heatmap(shift_A, shift_B, networks, save_path, label_A="A", label_B="B"):
    """Heatmap of per-network mean shift for two individual subjects."""
    from matplotlib.colors import LinearSegmentedColormap
    P = get_palette(); apply_style()
    net_names = list(networks.keys())
    nn = len(net_names)
    region_lists = [list(v[0]) for v in networks.values()]
    mean_A = np.array([shift_A[r].mean() for r in region_lists])
    mean_B = np.array([shift_B[r].mean() for r in region_lists])
    std_A  = np.array([shift_A[r].std()  for r in region_lists])
    std_B  = np.array([shift_B[r].std()  for r in region_lists])
    pooled = np.sqrt((std_A ** 2 + std_B ** 2) / 2)

    hm = np.column_stack([mean_A, mean_B])
    cmap = LinearSegmentedColormap.from_list("hm", ["white", "#B5D4F4", "#0D1B3E"])
    fig, ax = plt.subplots(figsize=(4.5, 6))
    im = ax.imshow(hm, cmap=cmap, aspect="auto", vmin=0, vmax=hm.max())
    plt.colorbar(im, ax=ax, label="mean $\\delta$", shrink=0.85)
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"Subj. {label_A}", f"Subj. {label_B}"], fontsize=9)
    ax.set_yticks(range(nn)); ax.set_yticklabels(net_names, fontsize=9)
    ax.set_title("Mean shift per network", fontsize=10, color=P["NAVY"])
    for ni in range(nn):
        diff = abs(mean_A[ni] - mean_B[ni])
        star = "*" if diff > 0.5 * pooled[ni] else ""
        for sj, val in enumerate([mean_A[ni], mean_B[ni]]):
            tc = "white" if val > hm.max() * 0.55 else P["NAVY"]
            label = f"{val:.2f}{star}" if sj == 0 else f"{val:.2f}"
            ax.text(sj, ni, label, ha="center", va="center", fontsize=8, color=tc)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    print(f"  {save_path}")
    return save_path


# ── group-level (S subjects per group) ────────────────────────────────────────
def plot_top_regions_shift(delta, significant, region_names, save_path,
                            label_0="group0", label_1="group1", top_n=20):
    """Top-N regions by |delta mean shift| between two groups, horizontal bars, FDR stars."""
    P = get_palette(); apply_style()
    top_idx = np.argsort(np.abs(delta))[::-1][:top_n]
    top_delta = delta[top_idx]
    top_names = [str(region_names[i]).replace("7Networks_", "").replace("_", " ") for i in top_idx]
    top_sig = significant[top_idx]
    colors_bar = [P["CORAL"] if d > 0 else P["BLUE"] for d in top_delta]
    y_pos = np.arange(top_n)[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.32 * top_n + 1.5))
    ax.barh(y_pos, top_delta, color=colors_bar, alpha=0.82, edgecolor=P["WHITE"], linewidth=0.5)
    for yi, (d, s) in enumerate(zip(top_delta, top_sig)):
        if s:
            ax.text(d + np.sign(d) * 0.02 * np.abs(delta).max(), y_pos[yi], "*",
                    ha="center", va="center", fontsize=11, color=P["NAVY"], fontweight="bold")
    ax.axvline(0, color=P["NAVY"], lw=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(top_names, fontsize=8)
    ax.set_xlabel(f"Δ mean shift ({label_1} − {label_0})", color=P["GRAY"])
    ax.set_title(f"Top-{top_n} regions by |Δ shift|  (* FDR p<0.05)", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=P["CORAL"], label=f"Higher shift in {label_1}"),
        Patch(facecolor=P["BLUE"],  label=f"Higher shift in {label_0}"),
    ], fontsize=8, loc="lower right", framealpha=0.85)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_shift_boxplot_by_network(shifts_0, shifts_1, rsn_ids, rsn_labels, save_path,
                                   label_0="group0", label_1="group1"):
    """Per-subject, per-network mean shift, violin group0 vs group1 for each network."""
    P = get_palette(); apply_style()
    n_rsn = len(rsn_labels)
    width = 0.35
    fig, ax = plt.subplots(figsize=(1.1 * n_rsn + 2, 5.5))
    for xi, rsn_id in enumerate(range(n_rsn)):
        mask_r = rsn_ids == rsn_id
        d0 = shifts_0[:, mask_r].mean(axis=1)
        d1 = shifts_1[:, mask_r].mean(axis=1)
        draw_violin(ax, d0, xi - width / 2, width * 0.9, P["BLUE"], P["NAVY"])
        draw_violin(ax, d1, xi + width / 2, width * 0.9, P["CORAL"], "darkred")
    ax.set_xticks(np.arange(n_rsn)); ax.set_xticklabels(rsn_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean shift per RSN (subjects)")
    ax.set_title(f"Shift per RSN: {label_0} vs {label_1}", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=P["BLUE"], alpha=0.65, label=label_0),
        Patch(facecolor=P["CORAL"], alpha=0.65, label=label_1),
    ], fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_shift_scatter_groups(mu_0, mu_1, delta, region_names, node_colors, save_path,
                               label_0="group0", label_1="group1", top_n=5):
    """Scatter of mean shift group0 vs group1 per region, colored by network."""
    P = get_palette(); apply_style()
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.scatter(mu_0, mu_1, c=node_colors, s=35, alpha=0.75, edgecolors=P["WHITE"], lw=0.4)
    lo = min(mu_0.min(), mu_1.min()) - 0.5
    hi = max(mu_0.max(), mu_1.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color=P["GRAY"], lw=1, label=f"{label_0} = {label_1}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    top_idx = np.argsort(delta)[::-1][:top_n]
    for ri in top_idx:
        ax.annotate(str(region_names[ri]).split("_")[-1], (mu_0[ri], mu_1[ri]),
                    fontsize=6.5, color=P["NAVY"], xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"Mean shift {label_0}", color=P["GRAY"])
    ax.set_ylabel(f"Mean shift {label_1}", color=P["GRAY"])
    ax.set_title(f"Region scatter (above diagonal = more shift in {label_1})", fontsize=9, color=P["NAVY"])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_significant_fraction_by_network(significant, rsn_ids, rsn_labels, save_path,
                                          label_0="group0", label_1="group1"):
    """% of regions per network with a significant (FDR<0.05) shift difference."""
    P = get_palette(); apply_style()
    n_rsn = len(rsn_labels)
    sig_per_rsn = np.array([significant[rsn_ids == r].sum() for r in range(n_rsn)])
    total_per_rsn = np.array([(rsn_ids == r).sum() for r in range(n_rsn)])
    frac_sig = sig_per_rsn / (total_per_rsn + 1e-9)
    rsn_cols = ["#7B1F9C", "#4682B4", "#2E7A2E", "#C43BFA", "#8BAF3A", "#E89020", "#CE3E50"]

    fig, ax = plt.subplots(figsize=(1.1 * n_rsn + 2, 5))
    ax.bar(np.arange(n_rsn), frac_sig * 100, color=(rsn_cols * ((n_rsn // 7) + 1))[:n_rsn],
           alpha=0.82, edgecolor=P["WHITE"], linewidth=0.5)
    for xi, (f, s, t) in enumerate(zip(frac_sig, sig_per_rsn, total_per_rsn)):
        ax.text(xi, f * 100 + 1, f"{s}/{t}", ha="center", fontsize=8, color=P["NAVY"])
    ax.set_xticks(np.arange(n_rsn)); ax.set_xticklabels(rsn_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% significant regions (FDR p<0.05)")
    ax.set_title(f"Fraction sig. regions per RSN  ({label_1} > {label_0})", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path
