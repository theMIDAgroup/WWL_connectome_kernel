"""Classification-result figures and publication tables.

plot_fold_accuracy_bars / plot_confusion_matrix / plot_lambda_c_per_fold split
WWL_full_pipeline.py's 3-panel plot_svm_results into atomic functions, driven
by the `diagnostics` dict crossval.nested_cv_kernel(..., return_diagnostics=True)
now returns (so this logic isn't reimplemented a second time).
"""

import os

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import matplotlib.pyplot as plt

from ..kernels import METHOD_LABELS, METHOD_ORDER
from ..style import apply_style, get_palette


def _stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def plot_accuracy_bars(results, save_path, p_val=None, chance=0.5, title="Balanced accuracy",
                        ylim=(0.3, 1.12), p_vals=None):
    """results: {method: (mean, std)}. If method names match METHOD_ORDER they're sorted/labeled accordingly.

    p_vals: optional {method: p_value} to star multiple bars individually
    (methods not in p_vals get no star). Takes precedence over p_val, which
    only ever stars bar 0 (kept for backward compatibility)."""
    P = get_palette(); apply_style()
    methods = [m for m in METHOD_ORDER if m in results] or list(results.keys())
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    palette_cycle = [P["BLUE"], P["TEAL"], P["AMBER"], P["CORAL"], P["GRAY"], P["VIOLET"], P["GREEN"]]
    colors = [palette_cycle[i % len(palette_cycle)] for i in range(len(methods))]

    x = np.arange(len(methods))
    mus = [results[m][0] for m in methods]
    sds = [results[m][1] for m in methods]

    fig, ax = plt.subplots(figsize=(1.3 * len(methods) + 2, 5.5))
    bars = ax.bar(x, mus, 0.6, color=colors, alpha=0.85,
                  yerr=sds, capsize=5, error_kw={"elinewidth": 1.5})
    bars[0].set_edgecolor(P["NAVY"]); bars[0].set_linewidth(2)
    ax.axhline(chance, color=P["GRAY"], lw=1.2, linestyle=":", label=f"Chance ({chance})")
    if p_vals is not None:
        for i, m in enumerate(methods):
            if m in p_vals:
                ax.text(i, mus[i] + sds[i] + 0.02, _stars(p_vals[m]), ha="center", va="bottom",
                        fontsize=15, color=P["NAVY"], fontweight="bold")
    elif p_val is not None:
        ax.text(0, mus[0] + sds[0] + 0.02, _stars(p_val), ha="center", va="bottom",
                fontsize=15, color=P["NAVY"], fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylim(*ylim)
    ax.set_title(title, fontsize=15, color=P["NAVY"], fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylabel("Balanced accuracy", color=P["GRAY"], fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.legend(fontsize=11, framealpha=0.7, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_h_selection(results, save_path, chance=0.5, best_h=None, title="WL depth (H) selection"):
    """
    Bar chart of nested-CV balanced accuracy per WL-propagation depth H, from
    crossval.select_best_h's `results` dict ({h: {"mean_bacc", "std_bacc"}}).
    The best H is outlined.
    """
    P = get_palette(); apply_style()
    h_values = sorted(results.keys())
    if best_h is None:
        best_h = max(h_values, key=lambda h: results[h]["mean_bacc"])
    mus = [results[h]["mean_bacc"] for h in h_values]
    sds = [results[h]["std_bacc"] for h in h_values]

    fig, ax = plt.subplots(figsize=(1.3 * len(h_values) + 2, 5))
    bars = ax.bar([str(h) for h in h_values], mus, 0.55, color=P["TEAL"], alpha=0.85,
                  yerr=sds, capsize=5, error_kw={"elinewidth": 1.5})
    best_idx = h_values.index(best_h)
    bars[best_idx].set_edgecolor(P["NAVY"]); bars[best_idx].set_linewidth(2.5)
    bars[best_idx].set_facecolor(P["BLUE"])
    ax.axhline(chance, color=P["GRAY"], lw=1.2, linestyle=":", label=f"Chance ({chance})")
    ax.set_xlabel("H (WL propagation iterations)", color=P["GRAY"])
    ax.set_ylabel("Balanced accuracy (nested CV)", color=P["GRAY"])
    ax.set_title(f"{title}\nbest H = {best_h}", fontsize=10, color=P["NAVY"], fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_permutation_null(null_dist, observed, p_val, save_path, title="Permutation test"):
    """Null-distribution histogram + observed accuracy + 95th percentile, for one model."""
    P = get_palette(); apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.hist(null_dist, bins=40, color=P["LGRAY"], edgecolor=P["GRAY"], density=True,
            alpha=0.8, label="Null distribution")
    ax.axvline(observed, color=P["BLUE"], lw=2.5, label=f"Observed={observed:.3f}")
    ax.axvline(np.percentile(null_dist, 95), color=P["CORAL"], lw=1.5, linestyle="--",
               label="95th pct (null)")
    ax.set_title(f"{title}  p={p_val:.4f} {_stars(p_val)}", fontsize=10, color=P["NAVY"])
    ax.set_xlabel("Balanced accuracy (permuted)", color=P["GRAY"])
    ax.set_ylabel("Density", color=P["GRAY"])
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_fold_accuracy_bars(per_fold_df, mean_acc, mean_bacc, save_path):
    """Per-fold accuracy / balanced-accuracy bars from a nested-CV diagnostics dict."""
    P = get_palette(); apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = per_fold_df["fold"].values
    ax.bar(x - 0.2, per_fold_df["acc"], 0.38, color=P["BLUE"], alpha=0.82, label="Accuracy")
    ax.bar(x + 0.2, per_fold_df["bacc"], 0.38, color=P["CORAL"], alpha=0.82, label="Balanced acc.")
    ax.axhline(mean_acc, color=P["BLUE"], lw=1.5, linestyle="--", label=f"mean acc={mean_acc:.3f}")
    ax.axhline(mean_bacc, color=P["CORAL"], lw=1.5, linestyle="--", label=f"mean bacc={mean_bacc:.3f}")
    ax.axhline(0.5, color=P["GRAY"], lw=1, linestyle=":")
    ax.set_ylim(0, 1.05); ax.set_xticks(x)
    ax.set_xlabel("Fold"); ax.set_ylabel("Score")
    ax.set_title("Per-fold scores", fontsize=10, color=P["NAVY"])
    ax.legend(fontsize=7.5, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_confusion_matrix(cm, class_names, save_path):
    """Row-normalized aggregate confusion matrix over all outer folds."""
    P = get_palette(); apply_style()
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("cm", [P["WHITE"], P["LBLUE"], P["BLUE"], P["NAVY"]])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, shrink=0.8)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            v = cm_norm[i, j]
            ax.text(j, i, f"{cm[i, j]}\n({v:.2f})", ha="center", va="center", fontsize=9,
                    color=P["WHITE"] if v > 0.55 else P["NAVY"])
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion matrix (aggregate)", fontsize=10, color=P["NAVY"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_lambda_c_per_fold(per_fold_df, save_path):
    """Calibrated lambda and best C per outer fold (dual log-scale y-axis)."""
    P = get_palette(); apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax2 = ax.twinx()
    folds = per_fold_df["fold"].values
    ax.plot(folds, per_fold_df["lambda"], "o-", color=P["TEAL"], lw=2, ms=7, label="λ (calibrated)")
    ax2.plot(folds, per_fold_df["best_C"], "s--", color=P["AMBER"], lw=2, ms=7, label="Best C")
    ax.set_xlabel("Fold"); ax.set_ylabel("λ", color=P["TEAL"])
    ax2.set_ylabel("C", color=P["AMBER"])
    ax.set_yscale("log"); ax2.set_yscale("log")
    ax.tick_params(axis="y", labelcolor=P["TEAL"])
    ax2.tick_params(axis="y", labelcolor=P["AMBER"])
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8, framealpha=0.85)
    ax.set_title("λ and C per fold", fontsize=10, color=P["NAVY"])
    ax.spines[["top"]].set_visible(False); ax2.spines[["top"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def build_csv_table(all_results, save_path):
    """all_results: list of dicts with scenario/p_val/p_mw/separation/results={method:(mean,std)}."""
    rows = []
    for res in all_results:
        p_mw = res.get("p_mw")
        sep = res.get("separation")
        row = {"scenario": res["scenario"],
               "p_perm": f"{res['p_val']:.4f}",
               "p_mw_distances": f"{p_mw:.4f}" if p_mw is not None else "n/a",
               "separation_d": f"{sep:.3f}" if sep is not None else "n/a"}
        for m in METHOD_ORDER:
            if m in res["results"]:
                mu, sd = res["results"][m]
                row[f"{m}_mean"] = f"{mu:.3f}"
                row[f"{m}_std"] = f"{sd:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    return df


def build_latex_table(all_results, save_path):
    """LaTeX table ready for paper insertion."""
    lines = [
        r"\begin{table}[ht]", r"\centering",
        r"\caption{Balanced accuracy (mean $\pm$ std) across difficulty"
        r" scenarios. $^{***}p<0.001$, $^{**}p<0.01$,"
        r" $^*p<0.05$ (permutation test).}",
        r"\label{tab:benchmark}", r"\small",
    ]
    methods = [m for m in METHOD_ORDER if m in all_results[0]["results"]]
    col_fmt = "l" + "c" * len(methods) + "c"
    lines.append(r"\begin{tabular}{" + col_fmt + r"}")
    lines.append(r"\toprule")
    header = "Scenario & " + " & ".join(
        METHOD_LABELS.get(m, m).replace("\n", " ") for m in methods) + r" & $p$-val \\"
    lines.append(header)
    lines.append(r"\midrule")

    for res in all_results:
        scen = res["scenario"].capitalize()
        p = res["p_val"]
        stars = (r"$^{***}$" if p < 0.001 else r"$^{**}$" if p < 0.01
                 else r"$^{*}$" if p < 0.05 else "n.s.")
        cells = []
        best = max(res["results"][m][0] for m in methods)
        for m in methods:
            mu, sd = res["results"][m]
            cell = f"{mu:.3f} $\\pm$ {sd:.3f}"
            if mu == best:
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)
        lines.append(f"{scen} & " + " & ".join(cells) + f" & {p:.4f}{stars} \\\\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    return save_path
