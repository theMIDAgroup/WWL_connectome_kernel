"""
Convenience "report" wrappers that lay out several of the atomic figures.*
panels into the classic combined multi-panel PNGs this project used to
produce by default (9-panel scenario figure, 4-panel population-shift
figure, ...). Kept only for backward-compatible one-shot output — the
default pipeline (wwl_benchmark.py) now calls the atomic figures.* functions
directly instead, so most benchmark_results/ output is one-figure-per-PNG.

Unlike figures.*, these draw multiple panels on one Figure inline (small
amount of duplication vs. figures.* is accepted here, deliberately, in
exchange for exact layout control) but always go through the shared
statistics helpers (shifts.compute_region_shift_stats, kernels.calibrate_lam*,
crossval.lambda_sensitivity_curve) rather than recomputing them.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

import matplotlib.pyplot as plt

from . import atlas as atlas_mod
from .crossval import lambda_sensitivity_curve
from .kernels import build_K, calibrate_lam_fisher, METHOD_LABELS, METHOD_ORDER
from .shifts import compute_region_shift_stats, region_shift_dataframe
from .style import apply_style, draw_violin, get_palette


def _intra_inter(D, n_each):
    S = len(D)
    intra_0 = [D[i, j] for i in range(n_each) for j in range(i + 1, n_each)]
    intra_1 = [D[i, j] for i in range(n_each, S) for j in range(i + 1, S)]
    inter = [D[i, j] for i in range(S) for j in range(i + 1, S)
             if (i < n_each) != (j < n_each)]
    return intra_0, intra_1, inter


def make_population_shift_report(name, shifts_0, shifts_1, atlas_df, out_dir,
                                  label_0="group0", label_1="group1"):
    """4-panel population-level shift figure + CSV (replaces wwl_benchmark.plot_population_shift
    and wwl_real_node_shift.plot_node_shift_ranking, which duplicated this exact layout)."""
    P = get_palette(); apply_style()
    region_names = atlas_df["region_name"].values
    rsn_ids = atlas_df["rsn_id"].values
    rsn_labels = atlas_df.drop_duplicates("rsn_id").sort_values("rsn_id")["rsn_label"].values
    N = len(region_names)
    n_rsn = len(rsn_labels)

    stats = compute_region_shift_stats(shifts_0, shifts_1)
    delta, significant = stats["delta"], stats["significant"]
    df_regions = region_shift_dataframe(stats, region_names, atlas_df["rsn_label"].values,
                                         label_0, label_1)
    csv_path = os.path.join(out_dir, f"{name}_shift_per_region.csv")
    df_regions.to_csv(csv_path, index=False)

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.40)
    fig.suptitle(f"Population-level embedding shift — {name}  "
                 f"(N={shifts_0.shape[0]} {label_0}, {shifts_1.shape[0]} {label_1})",
                 fontsize=13, fontweight="bold", color=P["NAVY"])

    # A — top-20 by |delta|
    ax = fig.add_subplot(gs[:, 0])
    top20 = np.argsort(np.abs(delta))[::-1][:20]
    top20_names = [str(r).replace("7Networks_", "").replace("_", " ") for r in region_names[top20]]
    colors_bar = [P["CORAL"] if d > 0 else P["BLUE"] for d in delta[top20]]
    y_pos = np.arange(20)[::-1]
    ax.barh(y_pos, delta[top20], color=colors_bar, alpha=0.82, edgecolor=P["WHITE"], linewidth=0.5)
    for yi, (d, s) in enumerate(zip(delta[top20], significant[top20])):
        if s:
            ax.text(d + np.sign(d) * 0.02 * np.abs(delta).max(), y_pos[yi], "*",
                    ha="center", va="center", fontsize=11, color=P["NAVY"], fontweight="bold")
    ax.axvline(0, color=P["NAVY"], lw=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(top20_names, fontsize=8)
    ax.set_xlabel(f"Δ mean shift ({label_1} − {label_0})", color=P["GRAY"])
    ax.set_title("A — Top-20 regions by |Δ shift|  (* FDR p<0.05)", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(facecolor=P["CORAL"], label=f"Higher shift in {label_1}"),
        mpatches.Patch(facecolor=P["BLUE"], label=f"Higher shift in {label_0}"),
    ], fontsize=8, loc="lower right", framealpha=0.85)

    # B — per-RSN violin
    ax = fig.add_subplot(gs[0, 1:])
    width = 0.35
    for xi, rsn_id in enumerate(range(n_rsn)):
        mask_r = rsn_ids == rsn_id
        d0 = shifts_0[:, mask_r].mean(axis=1)
        d1 = shifts_1[:, mask_r].mean(axis=1)
        draw_violin(ax, d0, xi - width / 2, width * 0.9, P["BLUE"], P["NAVY"])
        draw_violin(ax, d1, xi + width / 2, width * 0.9, P["CORAL"], "darkred")
    ax.set_xticks(np.arange(n_rsn)); ax.set_xticklabels(rsn_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean shift per RSN (subjects)")
    ax.set_title(f"B — Per-RSN shift distribution: {label_0} vs {label_1}", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(facecolor=P["BLUE"], alpha=0.65, label=label_0),
        mpatches.Patch(facecolor=P["CORAL"], alpha=0.65, label=label_1),
    ], fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)

    # C — scatter mean shift
    ax = fig.add_subplot(gs[1, 1])
    node_cols = [atlas_mod.REGION_COLOR.get(i, P["GRAY"]) for i in range(N)]
    ax.scatter(stats["mu_0"], stats["mu_1"], c=node_cols, s=35, alpha=0.75, edgecolors=P["WHITE"], lw=0.4)
    lo = min(stats["mu_0"].min(), stats["mu_1"].min()) - 0.5
    hi = max(stats["mu_0"].max(), stats["mu_1"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color=P["GRAY"], lw=1, label=f"{label_0} = {label_1}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for ri in np.argsort(delta)[::-1][:5]:
        ax.annotate(region_names[ri].split("_")[-1], (stats["mu_0"][ri], stats["mu_1"][ri]),
                    fontsize=6.5, color=P["NAVY"], xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(f"Mean shift {label_0}", color=P["GRAY"])
    ax.set_ylabel(f"Mean shift {label_1}", color=P["GRAY"])
    ax.set_title(f"C — Region scatter (above diagonal = more shift in {label_1})",
                 fontsize=10, color=P["NAVY"])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)

    # D — % significant per RSN
    ax = fig.add_subplot(gs[1, 2])
    sig_per_rsn = np.array([significant[rsn_ids == r].sum() for r in range(n_rsn)])
    total_per_rsn = np.array([(rsn_ids == r).sum() for r in range(n_rsn)])
    frac_sig = sig_per_rsn / (total_per_rsn + 1e-9)
    rsn_cols = ["#7B1F9C", "#4682B4", "#2E7A2E", "#C43BFA", "#8BAF3A", "#E89020", "#CE3E50"]
    ax.bar(np.arange(n_rsn), frac_sig * 100, color=(rsn_cols * ((n_rsn // 7) + 1))[:n_rsn],
           alpha=0.82, edgecolor=P["WHITE"], linewidth=0.5)
    for xi, (f, s, t) in enumerate(zip(frac_sig, sig_per_rsn, total_per_rsn)):
        ax.text(xi, f * 100 + 1, f"{s}/{t}", ha="center", fontsize=8, color=P["NAVY"])
    ax.set_xticks(np.arange(n_rsn)); ax.set_xticklabels(rsn_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("% significant regions (FDR p<0.05)")
    ax.set_title(f"D — Fraction sig. regions per RSN ({label_1} > {label_0})", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, f"{name}_population_shift.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return fig_path, csv_path, df_regions


def make_distance_lambda_report(name, D, y, n_each, out_dir, Z_emb=None, ev_emb=None):
    """4-panel distance/lambda figure (replaces wwl_benchmark.plot_distance_lambda)."""
    P = get_palette(); apply_style()
    S = len(y)
    intra_0, intra_1, inter = _intra_inter(D, n_each)
    lam_fi, lam_mu = calibrate_lam_fisher(D, (np.arange(S) >= n_each).astype(int))
    _, p_mw = mannwhitneyu(intra_0 + intra_1, inter, alternative="less")
    stars = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(f"Distance space analysis — {name}  (N={n_each}+{n_each})",
                 fontsize=12, fontweight="bold", color=P["NAVY"])

    ax = axes[0]
    cmap_d = LinearSegmentedColormap.from_list("dist", [P["WHITE"], P["LBLUE"], P["NAVY"]])
    im = ax.imshow(D, cmap=cmap_d, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="W1 distance")
    ax.axhline(n_each - 0.5, color=P["CORAL"], lw=1.5); ax.axvline(n_each - 0.5, color=P["CORAL"], lw=1.5)
    ax.set_title("A — W1 distance matrix", fontsize=10, color=P["NAVY"])
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    bp = ax.boxplot([intra_0, intra_1, inter], patch_artist=True, labels=["G0", "G1", "G0–G1"])
    for patch, c in zip(bp["boxes"], [P["BLUE"], P["CORAL"], P["GRAY"]]):
        patch.set_facecolor(c); patch.set_alpha(0.6)
    ax.set_ylabel("$W_1$ distance")
    ax.set_title("B — Distance distributions by group", fontsize=10, color=P["NAVY"])
    ax.grid(True, alpha=0.25); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    d_all = intra_0 + intra_1 + inter
    bins = np.linspace(min(d_all), max(d_all), 35)
    for data, lbl, col in [(intra_0, f"intra-G0  μ={np.mean(intra_0):.2f}", P["BLUE"]),
                            (intra_1, f"intra-G1  μ={np.mean(intra_1):.2f}", P["CORAL"]),
                            (inter, f"inter     μ={np.mean(inter):.2f}", P["GRAY"])]:
        ax.hist(data, bins=bins, alpha=0.55, color=col, density=True, label=lbl)
        ax.axvline(np.mean(data), color=col, lw=1.8, linestyle="--")
    ax.set_xlabel("$W_1$ distance", fontsize=9); ax.legend(fontsize=8, framealpha=0.85)
    ax.set_title(f"C — Distance distributions  {stars} (p={p_mw:.3f})", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.22)

    ax = axes[3]
    lam_grid, accs = lambda_sensitivity_curve(D, y, lam_mu=lam_mu)
    best_lam = lam_grid[np.argmax(accs)]
    ax.semilogx(lam_grid, accs, color=P["TEAL"], lw=2, zorder=3)
    ax.axvline(lam_mu, color=P["AMBER"], lw=1.8, linestyle="--", label=f"1/μ (λ={lam_mu:.3f})", zorder=4)
    ax.axvline(lam_fi, color=P["CORAL"], lw=1.8, linestyle=":", label=f"Fisher (λ={lam_fi:.3f})", zorder=4)
    ax.axvline(best_lam, color=P["NAVY"], lw=1.8, linestyle="-", label=f"CV-best (λ={best_lam:.3f})", zorder=4)
    ax.axhline(0.5, color=P["LGRAY"], lw=1, linestyle=":")
    ax.fill_between(lam_grid, accs, 0.5, where=accs > 0.5, alpha=0.12, color=P["TEAL"])
    ax.set_xlabel("λ (log scale)", fontsize=9); ax.set_ylabel("Bal. accuracy (3-fold CV)", fontsize=9)
    ax.set_title("D — λ sensitivity", fontsize=10, color=P["NAVY"])
    ax.legend(fontsize=7.5, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.22)
    ax.set_ylim(0.35, 1.05); ax.set_xlim(lam_grid[0], lam_grid[-1])

    fig.tight_layout()
    save_path = os.path.join(out_dir, f"{name}_distance_lambda.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path, best_lam


def make_distance_analysis_report(name, D, y, n_each, out_dir, Z_emb=None, ev_emb=None):
    """5-panel distance-matrix analysis figure (replaces wwl_benchmark.plot_distance_matrix_analysis)."""
    P = get_palette(); apply_style()
    S = len(y)
    lam = 1.0 / (D[D > 0].mean() + 1e-9)
    K = build_K(D, lam)
    intra_0, intra_1, inter = _intra_inter(D, n_each)
    _, p_mw = mannwhitneyu(intra_0 + intra_1, inter, alternative="less")
    stars = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."

    fig = plt.figure(figsize=(20, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.15], hspace=0.40, wspace=0.35)
    fig.suptitle(f"Distance matrix analysis — {name}  (N={n_each}+{n_each}, λ={lam:.4f})",
                 fontsize=12, fontweight="bold", color=P["NAVY"])

    ax = fig.add_subplot(gs[0, 0])
    cmap_d = LinearSegmentedColormap.from_list("dist", [P["WHITE"], P["LBLUE"], P["NAVY"]])
    im = ax.imshow(D, cmap=cmap_d, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="W1 distance")
    ax.axhline(n_each - 0.5, color=P["CORAL"], lw=1.5); ax.axvline(n_each - 0.5, color=P["CORAL"], lw=1.5)
    ax.set_title("A — W1 distance matrix", fontsize=9, color=P["NAVY"])
    ax.set_xlabel("Subject"); ax.set_ylabel("Subject")

    ax = fig.add_subplot(gs[0, 1])
    cmap_k = LinearSegmentedColormap.from_list("kern", [P["WHITE"], P["LBLUE"], P["BLUE"], P["NAVY"]])
    im = ax.imshow(K, cmap=cmap_k, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8, label="kernel similarity")
    ax.axhline(n_each - 0.5, color=P["CORAL"], lw=1.5); ax.axvline(n_each - 0.5, color=P["CORAL"], lw=1.5)
    ax.set_title(f"B — Kernel matrix (λ={lam:.3f})", fontsize=9, color=P["NAVY"])
    ax.set_xlabel("Subject"); ax.set_ylabel("Subject")

    ax = fig.add_subplot(gs[0, 2])
    d_all = intra_0 + intra_1 + inter
    bins = np.linspace(min(d_all), max(d_all), 40)
    for data, lbl, col in [(intra_0, f"intra-G0  μ={np.mean(intra_0):.2f}", P["BLUE"]),
                            (intra_1, f"intra-G1  μ={np.mean(intra_1):.2f}", P["CORAL"]),
                            (inter, f"inter     μ={np.mean(inter):.2f}", P["GRAY"])]:
        ax.hist(data, bins=bins, alpha=0.55, color=col, density=True, label=lbl)
        ax.axvline(np.mean(data), color=col, lw=1.8, linestyle="--")
    ax.set_xlabel("W1 distance"); ax.set_ylabel("Density"); ax.legend(fontsize=7.5, framealpha=0.85)
    ax.set_title(f"C — Distance distributions | MW intra<inter: {stars} (p={p_mw:.4f})",
                 fontsize=9, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)

    ax = fig.add_subplot(gs[1, :2])
    if Z_emb is not None:
        Z_e, ev_e = Z_emb, ev_emb
    else:
        Z_e, ev_e = np.zeros((S, 2)), [0.5, 0.3]
    for g, sl, col in [("G0", slice(None, n_each), P["BLUE"]), ("G1", slice(n_each, None), P["CORAL"])]:
        ax.scatter(Z_e[sl, 0], Z_e[sl, 1], c=col, label=g, s=55, alpha=0.8, edgecolors=P["WHITE"], lw=0.5)
        ax.scatter(*Z_e[sl].mean(0), s=200, c=col, marker="*", zorder=6, edgecolors=P["NAVY"], lw=1.2)
    c0, c1 = Z_e[:n_each].mean(0), Z_e[n_each:].mean(0)
    sp = np.sqrt((Z_e[:n_each].var(0).mean() + Z_e[n_each:].var(0).mean()) / 2)
    sep_r = np.linalg.norm(c0 - c1) / (sp + 1e-9)
    ax.set_xlabel(f"PC1 ({ev_e[0]*100:.1f}%)"); ax.set_ylabel(f"PC2 ({ev_e[1]*100:.1f}%)")
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_title(f"D — PCA of joint embeddings  (sep={sep_r:.2f})", fontsize=9, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 2:])
    ax.axis("off")
    rows = [
        ["", "Mean", "Std", "Median", "IQR"],
        ["intra-G0", f"{np.mean(intra_0):.3f}", f"{np.std(intra_0):.3f}", f"{np.median(intra_0):.3f}",
         f"{np.percentile(intra_0,75)-np.percentile(intra_0,25):.3f}"],
        ["intra-G1", f"{np.mean(intra_1):.3f}", f"{np.std(intra_1):.3f}", f"{np.median(intra_1):.3f}",
         f"{np.percentile(intra_1,75)-np.percentile(intra_1,25):.3f}"],
        ["inter-group", f"{np.mean(inter):.3f}", f"{np.std(inter):.3f}", f"{np.median(inter):.3f}",
         f"{np.percentile(inter,75)-np.percentile(inter,25):.3f}"],
        ["", "", "", "", ""],
        ["Separation index",
         f"{(np.mean(inter)-np.mean(intra_0+intra_1))/(np.std(intra_0+intra_1)+1e-9):.3f}",
         "", "(Cohen's d approx)", ""],
        ["Mann-Whitney U", f"p={p_mw:.4f}", stars, "(intra < inter)", ""],
    ]
    col_widths = [0.22, 0.16, 0.16, 0.16, 0.16]
    for ri, row in enumerate(rows):
        for ci, cell in enumerate(row):
            x_pos = sum(col_widths[:ci]) + 0.02
            weight = "bold" if ri == 0 or ci == 0 else "normal"
            color = (P["NAVY"] if ri == 0 else P["BLUE"] if ri == 1 else P["CORAL"] if ri == 2 else P["GRAY"])
            ax.text(x_pos, 1 - ri * 0.13, cell, transform=ax.transAxes,
                    fontsize=9.5, fontweight=weight, color=color, va="top")
    ax.set_title("E — Distance statistics", fontsize=9, color=P["NAVY"])

    fig.tight_layout()
    save_path = os.path.join(out_dir, f"{name}_distance_analysis.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path, p_mw, float(np.mean(inter) - np.mean(intra_0 + intra_1))


def report_accuracy_comparison(all_results, save_path):
    """Multi-scenario accuracy grid (replaces wwl_benchmark.plot_accuracy_comparison)."""
    P = get_palette(); apply_style()
    n_scen = len(all_results)
    fig, axes = plt.subplots(1, n_scen, figsize=(4.5 * n_scen, 5.5), sharey=True)
    if n_scen == 1: axes = [axes]
    fig.suptitle("Balanced accuracy — WWL vs baselines", fontsize=13, fontweight="bold",
                 color=P["NAVY"], y=1.02)
    palette_cycle = [P["BLUE"], P["TEAL"], P["AMBER"], P["CORAL"], P["GRAY"], P["VIOLET"], P["GREEN"]]

    for ax, res in zip(axes, all_results):
        methods = [m for m in METHOD_ORDER if m in res["results"]]
        colors = [palette_cycle[i % len(palette_cycle)] for i in range(len(methods))]
        x = np.arange(len(methods))
        mus = [res["results"][m][0] for m in methods]
        sds = [res["results"][m][1] for m in methods]
        bars = ax.bar(x, mus, 0.6, color=colors, alpha=0.85,
                      yerr=sds, capsize=5, error_kw={"elinewidth": 1.5})
        bars[0].set_edgecolor(P["NAVY"]); bars[0].set_linewidth(2)
        ax.axhline(0.5, color=P["GRAY"], lw=1.2, linestyle=":", label="Chance (0.5)")
        stars = "***" if res["p_val"] < 0.001 else "**" if res["p_val"] < 0.01 else \
                "*" if res["p_val"] < 0.05 else "n.s."
        ax.text(0, mus[0] + sds[0] + 0.02, stars, ha="center", va="bottom",
                fontsize=12, color=P["NAVY"], fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods], fontsize=8.5)
        ax.set_ylim(0.3, 1.12)
        ax.set_title(res["scenario"], fontsize=11, color=P["NAVY"], fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Balanced accuracy", color=P["GRAY"])
        ax.legend(fontsize=8, framealpha=0.7, loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def make_scenario_report(name, embs_0, embs_1, D, n_each, shifts_0, shifts_1, atlas_df, out_dir,
                          label_0="group0", label_1="group1"):
    """9-panel per-scenario master figure (replaces make_scenario_figures.make_scenario_figure)."""
    P = get_palette(); apply_style()
    S = 2 * n_each
    region_names = atlas_df["region_name"].values
    rsn_ids = atlas_df["rsn_id"].values
    rsn_labels = atlas_df.drop_duplicates("rsn_id").sort_values("rsn_id")["rsn_label"].values
    n_rsn = len(rsn_labels)

    X0 = np.array([e.ravel() for e in embs_0])
    X1 = np.array([e.ravel() for e in embs_1])
    intra_0, intra_1, inter = _intra_inter(D, n_each)
    _, p_mw = mannwhitneyu(intra_0 + intra_1, inter, alternative="less")
    stars = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."

    stats = compute_region_shift_stats(shifts_0, shifts_1)
    delta, significant = stats["delta"], stats["significant"]

    fig = plt.figure(figsize=(20, 18))
    fig.suptitle(f"WWL connectome kernel — scenario: {name.upper()}  (N={n_each}+{n_each})",
                 fontsize=14, fontweight="bold", color=P["NAVY"], y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35,
                           top=0.95, bottom=0.04, left=0.06, right=0.97)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    for ax, X, col, lbl, tag in [(axes[0], X0, P["BLUE"], label_0, "A"), (axes[1], X1, P["CORAL"], label_1, "B")]:
        Z = PCA(n_components=2).fit(X).transform(X)
        ev = PCA(n_components=2).fit(X).explained_variance_ratio_
        ax.scatter(Z[:, 0], Z[:, 1], s=35, c=col, alpha=0.75, edgecolors="white", linewidth=0.35)
        ax.set_xlabel(f"PC1 ({100*ev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({100*ev[1]:.1f}%)")
        ax.set_title(f"{tag} — PCA of {lbl} subjects")

    cmap_d = LinearSegmentedColormap.from_list("d", [P["WHITE"], P["LBLUE"], P["NAVY"]])
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(D, cmap=cmap_d, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.82, label="$W_1$ distance")
    ax.axhline(n_each - 0.5, color=P["CORAL"], lw=1.5); ax.axvline(n_each - 0.5, color=P["CORAL"], lw=1.5)
    ax.set_title("C — $W_1$ distance matrix\n(all subjects)", fontsize=10, color=P["NAVY"])
    ax.set_xticks([]); ax.set_yticks([])

    ax = fig.add_subplot(gs[1, 0])
    d_all = intra_0 + intra_1 + inter
    bins = np.linspace(min(d_all), max(d_all), 35)
    for data, lbl, col in [(intra_0, f"intra-{label_0}  μ={np.mean(intra_0):.2f}", P["BLUE"]),
                            (intra_1, f"intra-{label_1}  μ={np.mean(intra_1):.2f}", P["CORAL"]),
                            (inter, f"inter-group  μ={np.mean(inter):.2f}", P["GRAY"])]:
        ax.hist(data, bins=bins, alpha=0.55, color=col, density=True, label=lbl)
        ax.axvline(np.mean(data), color=col, lw=2, linestyle="--")
    ax.set_xlabel("$W_1$ distance", fontsize=9); ax.set_ylabel("Density", fontsize=9)
    ax.legend(fontsize=8.5, framealpha=0.88)
    ax.set_title(f"D — Distance distributions\nMann-Whitney intra<inter: {stars} (p={p_mw:.3f})",
                 fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.22)

    ax = axes[4]
    lam = 1.0 / (D[D > 0].mean() + 1e-9)
    K = build_K(D, lam)
    im = ax.imshow(K, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.7, pad=0.04)
    ax.axhline(n_each - 0.5, color="white", linestyle="--", linewidth=1, alpha=0.7)
    ax.axvline(n_each - 0.5, color="white", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xticks([n_each // 2, n_each + n_each // 2]); ax.set_xticklabels([label_0, label_1], fontsize=8, fontweight="bold")
    ax.set_yticks([n_each // 2, n_each + n_each // 2]); ax.set_yticklabels([label_0, label_1], fontsize=8, fontweight="bold")
    ax.set_title("E — WWL Kernel Matrix (Similarity)", fontsize=10, fontweight="bold", color=P["NAVY"])

    ax = axes[5]
    X_joint = np.vstack([X0, X1])
    pca_j = PCA(n_components=2, random_state=42)
    Z_j = pca_j.fit_transform(X_joint)
    ev_j = pca_j.explained_variance_ratio_
    for g, sl, col in [(label_0, slice(None, n_each), P["BLUE"]), (label_1, slice(n_each, None), P["CORAL"])]:
        ax.scatter(Z_j[sl, 0], Z_j[sl, 1], c=col, label=g, s=45, alpha=0.72, edgecolors="white", lw=0.35, zorder=3)
        ax.scatter(*Z_j[sl].mean(0), s=220, c=col, marker="*", zorder=6, edgecolors=P["NAVY"], lw=1.2)
    c0, c1 = Z_j[:n_each].mean(0), Z_j[n_each:].mean(0)
    sp = np.sqrt((Z_j[:n_each].var(0).mean() + Z_j[n_each:].var(0).mean()) / 2)
    sep = np.linalg.norm(c0 - c1) / (sp + 1e-9)
    ax.set_xlabel(f"PC1 ({ev_j[0]*100:.1f}%)", fontsize=9); ax.set_ylabel(f"PC2 ({ev_j[1]*100:.1f}%)", fontsize=9)
    ax.set_title(f"F — PCA joint embedding space\nsep={sep:.2f}", fontsize=10, color=P["NAVY"])
    ax.legend(fontsize=9, framealpha=0.85); ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.22)

    ax = axes[6]
    top20 = np.argsort(np.abs(delta))[::-1][:20]
    top20_n = [str(region_names[i]).replace("7Networks_", "").replace("_", " ") for i in top20]
    colors_b = [P["CORAL"] if d > 0 else P["BLUE"] for d in delta[top20]]
    y_pos = np.arange(20)[::-1]
    ax.barh(y_pos, delta[top20], color=colors_b, alpha=0.82, edgecolor="white", linewidth=0.4)
    for yi, (d, s) in enumerate(zip(delta[top20], significant[top20])):
        if s:
            ax.text(d + np.sign(d) * 0.015 * np.abs(delta).max(), y_pos[yi], "*", ha="center", va="center",
                    fontsize=11, color=P["NAVY"], fontweight="bold")
    ax.axvline(0, color=P["NAVY"], lw=0.8)
    ax.set_yticks(y_pos); ax.set_yticklabels(top20_n, fontsize=7.5)
    ax.set_xlabel(f"Δ mean shift ({label_1} − {label_0})", fontsize=9)
    ax.set_title("G — Top-20 regions by |Δδ|\n(* FDR q<0.05)", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[mpatches.Patch(facecolor=P["CORAL"], label=f"Higher shift in {label_1}"),
                        mpatches.Patch(facecolor=P["BLUE"], label=f"Higher shift in {label_0}")],
              fontsize=8, loc="lower right", framealpha=0.85)

    ax = axes[7]
    width = 0.33
    for xi, rsn_id in enumerate(range(n_rsn)):
        mask = rsn_ids == rsn_id
        for data, off, col, mc in [(shifts_0[:, mask].mean(axis=1), -width/2, P["BLUE"], P["NAVY"]),
                                    (shifts_1[:, mask].mean(axis=1), width/2, P["CORAL"], "darkred")]:
            draw_violin(ax, data, xi + off, width * 0.9, col, mc)
    ax.set_xticks(np.arange(n_rsn)); ax.set_xticklabels(rsn_labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Mean shift per RSN (subjects)", fontsize=9)
    ax.set_title(f"H — Embedding shift per RSN\n({label_0} blue, {label_1} orange)", fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", alpha=0.28)
    ax.legend(handles=[mpatches.Patch(facecolor=P["BLUE"], alpha=0.65, label=label_0),
                        mpatches.Patch(facecolor=P["CORAL"], alpha=0.65, label=label_1)],
              fontsize=9, framealpha=0.85)

    ax = axes[8]
    node_cols = [atlas_mod.REGION_COLOR.get(i, P["GRAY"]) for i in range(len(region_names))]
    ax.scatter(stats["mu_0"], stats["mu_1"], c=node_cols, s=40, alpha=0.78, edgecolors="white", lw=0.35, zorder=3)
    lo = min(stats["mu_0"].min(), stats["mu_1"].min()) - 0.5
    hi = max(stats["mu_0"].max(), stats["mu_1"].max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color=P["GRAY"], lw=1.2, label=f"{label_0} = {label_1}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for ri in np.argsort(delta)[::-1][:5]:
        ax.annotate(str(region_names[ri]).split("_")[-1], (stats["mu_0"][ri], stats["mu_1"][ri]),
                    fontsize=7, color=P["NAVY"], xytext=(5, 4), textcoords="offset points", fontweight="bold")
    ax.legend(handles=atlas_mod.legend_handles(), fontsize=7, framealpha=0.85, ncol=2,
              title="RSN", title_fontsize=7.5)
    ax.set_xlabel(f"Mean shift — {label_0}", fontsize=9); ax.set_ylabel(f"Mean shift — {label_1}", fontsize=9)
    ax.set_title(f"I — Region-level shift: {label_0} vs {label_1}\n(above diagonal = more shift in {label_1})",
                 fontsize=10, color=P["NAVY"])
    ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.22)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fig_scenario_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return path


def make_comparison_report(scenario_data, out_dir, scenario_labels=None):
    """
    2-row multi-scenario comparison figure (replaces make_scenario_figures.make_comparison_figure).
    scenario_data: {scenario_name: {"D":..., "n_each":..., "X_cn":..., "X_ad":...}} (X_cn/X_ad =
    flattened embeddings for the two groups, kept as X_cn/X_ad for call-site compatibility).
    """
    P = get_palette(); apply_style()
    scenarios = list(scenario_data.keys())
    n_scen = len(scenarios)
    fig, axes = plt.subplots(2, n_scen, figsize=(5 * n_scen, 10))
    if n_scen == 1:
        axes = axes.reshape(2, 1)
    fig.suptitle("Multi-scenario comparison — WWL kernel", fontsize=13, fontweight="bold",
                 color=P["NAVY"], y=1.01)
    scenario_labels = scenario_labels or {}

    for col, scen in enumerate(scenarios):
        d = scenario_data[scen]
        X0, X1, D, n_each = d["X_cn"], d["X_ad"], d["D"], d["n_each"]
        intra_0, intra_1, inter = _intra_inter(D, n_each)
        intra = intra_0 + intra_1
        _, p_mw = mannwhitneyu(intra, inter, alternative="less")
        stars = "***" if p_mw < 0.001 else "**" if p_mw < 0.01 else "*" if p_mw < 0.05 else "n.s."

        ax = axes[0, col]
        Z = PCA(n_components=2, random_state=42).fit_transform(np.vstack([X0, X1]))
        for g, sl, col_c in [("G0", slice(None, n_each), P["BLUE"]), ("G1", slice(n_each, None), P["CORAL"])]:
            ax.scatter(Z[sl, 0], Z[sl, 1], c=col_c, label=g, s=35, alpha=0.68, edgecolors="white", lw=0.3, zorder=3)
            ax.scatter(*Z[sl].mean(0), s=200, c=col_c, marker="*", zorder=6, edgecolors=P["NAVY"], lw=1.2)
        c0, c1 = Z[:n_each].mean(0), Z[n_each:].mean(0)
        sp = np.sqrt((Z[:n_each].var(0).mean() + Z[n_each:].var(0).mean()) / 2)
        sep = np.linalg.norm(c0 - c1) / (sp + 1e-9)
        if col == 0: ax.set_ylabel("PC2", fontsize=9)
        ax.set_title(f"{scenario_labels.get(scen, scen)}\nsep={sep:.2f}", fontsize=10,
                     color=P["NAVY"], fontweight="bold")
        ax.legend(fontsize=8, framealpha=0.85); ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.22)

        ax = axes[1, col]
        d_all = intra + inter
        bins = np.linspace(min(d_all), max(d_all), 30)
        for data, lbl, color in [(intra, f"intra μ={np.mean(intra):.1f}", P["BLUE"]),
                                  (inter, f"inter μ={np.mean(inter):.1f}", P["GRAY"])]:
            ax.hist(data, bins=bins, alpha=0.55, color=color, density=True, label=lbl)
            ax.axvline(np.mean(data), color=color, lw=1.8, linestyle="--")
        if col == 0: ax.set_ylabel("Density", fontsize=9)
        ax.set_title(f"{stars}  p={p_mw:.3f}", fontsize=10, color=P["NAVY"])
        ax.legend(fontsize=8, framealpha=0.85); ax.spines[["top", "right"]].set_visible(False); ax.grid(True, alpha=0.22)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig_scenario_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return path


def report_permutation_grid(all_results, save_path):
    """Multi-scenario permutation-null grid (replaces wwl_benchmark.plot_permutation)."""
    P = get_palette(); apply_style()
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5))
    if n == 1: axes = [axes]
    fig.suptitle("Permutation test — WWL kernel", fontsize=13, fontweight="bold", color=P["NAVY"], y=1.02)

    for ax, res in zip(axes, all_results):
        null = res["null_dist"]; obs = res["results"]["WWL"][0]; p = res["p_val"]
        ax.hist(null, bins=40, color=P["LGRAY"], edgecolor=P["GRAY"], density=True, alpha=0.8,
                label="Null distribution")
        ax.axvline(obs, color=P["BLUE"], lw=2.5, label=f"Observed={obs:.3f}")
        ax.axvline(np.percentile(null, 95), color=P["CORAL"], lw=1.5, linestyle="--", label="95th pct (null)")
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"{res['scenario']}  p={p:.4f} {stars}", fontsize=10, color=P["NAVY"])
        ax.set_xlabel("Balanced accuracy (permuted)", color=P["GRAY"])
        ax.set_ylabel("Density", color=P["GRAY"])
        ax.legend(fontsize=8, framealpha=0.85)
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path
