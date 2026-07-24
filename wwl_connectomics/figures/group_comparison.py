"""N-group PCA comparison on flattened embeddings — generalizes the PCA panel
that used to be duplicated across make_scenario_figure, make_comparison_figure
and wwl_real_group_comparison.make_group_comparison_figure (each hardcoded to
2 groups named CN/AD). For the intra/inter distance histogram counterpart see
figures.distance_plots.plot_distance_intra_inter_hist — deliberately not
reimplemented here to avoid the exact duplication this module set out to fix.
"""

import numpy as np
from matplotlib.patches import Ellipse

import matplotlib.pyplot as plt

from ..style import apply_style, get_palette


def _draw_confidence_ellipse(ax, points_2d, color, confidence=0.95, **kwargs):
    """
    Draw a covariance-based confidence ellipse for a 2D point cloud (e.g. one
    group's PCA/LDA scores), assuming an approximately bivariate-normal
    spread. Radius scaled via the chi-square quantile for `confidence` (df=2),
    so confidence=0.95 is an honest ~95% coverage ellipse, not just "2 std".
    Silently skips groups with too few points to estimate a covariance.
    """
    from scipy.stats import chi2

    if len(points_2d) < 3:
        return
    cov = np.cov(points_2d, rowvar=False)
    if not np.all(np.isfinite(cov)):
        return
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0, None)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    scale = np.sqrt(chi2.ppf(confidence, df=2))
    width, height = 2 * scale * np.sqrt(eigvals)
    mean = points_2d.mean(axis=0)
    ell = Ellipse(mean, width=width, height=height, angle=angle,
                  facecolor=color, edgecolor=color, **kwargs)
    ax.add_patch(ell)


def plot_group_pca(X, group_sizes, save_path, title=None, seed=42, confidence=0.95):
    """
    X: (S, D) flattened embeddings, rows ordered to match group_sizes (all of
    group 1 first, then group 2, ...). group_sizes: OrderedDict name->n.
    Draws centroids (stars) and a `confidence`-coverage ellipse per group
    (skipped for groups with <3 points); if exactly 2 groups, annotates a
    separation index. Returns (Z, explained_variance_ratio, separation_or_None).
    """
    from sklearn.decomposition import PCA

    P = get_palette(); apply_style()
    cats = list(group_sizes.keys())
    n_cats = len(cats)
    bounds = np.cumsum([0] + list(group_sizes.values()))
    slices = {cat: slice(bounds[i], bounds[i + 1]) for i, cat in enumerate(cats)}

    pca = PCA(n_components=2, random_state=seed)
    Z = pca.fit_transform(X)
    ev = pca.explained_variance_ratio_

    if n_cats <= 2:
        colors_cycle = [P["BLUE"], P["CORAL"]]
    else:
        cmap = plt.get_cmap("tab10" if n_cats <= 10 else "tab20")
        colors_cycle = [cmap(i % cmap.N) for i in range(n_cats)]

    fig, ax = plt.subplots(figsize=(6, 5.5))
    centroids = {}
    for i, cat in enumerate(cats):
        sl = slices[cat]
        col = colors_cycle[i]
        _draw_confidence_ellipse(ax, Z[sl], col, confidence=confidence,
                                  alpha=0.12, lw=1.5, linestyle="--", zorder=1)
        ax.scatter(Z[sl, 0], Z[sl, 1], c=[col], label=f"{cat} (n={group_sizes[cat]})",
                   s=40, alpha=0.72, edgecolors="white", lw=0.35, zorder=3)
        c = Z[sl].mean(axis=0)
        centroids[cat] = c
        ax.scatter(*c, s=220, c=[col], marker="*", zorder=6, edgecolors=P["NAVY"], lw=1.2)

    sep = None
    title_extra = ""
    if n_cats == 2:
        c0, c1 = centroids[cats[0]], centroids[cats[1]]
        sl0, sl1 = slices[cats[0]], slices[cats[1]]
        sp = np.sqrt((Z[sl0].var(0).mean() + Z[sl1].var(0).mean()) / 2)
        sep = float(np.linalg.norm(c0 - c1) / (sp + 1e-9))
        title_extra = f"\nsep={sep:.2f}"

    ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)", fontsize=9)
    ax.set_title((title or "PCA of embeddings") + title_extra, fontsize=10,
                 color=P["NAVY"], fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, alpha=0.22)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return Z, ev, sep


def plot_lda_projection(X, group_sizes, save_path, title=None, cv_splits=5, seed=42):
    """
    Supervised (label-aware) projection, complementary to plot_group_pca.

    PCA shows the directions of maximum TOTAL variance, which in noisy
    high-dimensional embeddings are often dominated by per-subject noise
    rather than the group signal — a classifier can separate groups well
    even when the PCA plot looks flat. LDA instead finds the direction(s)
    that maximize between-group vs within-group variance directly from the
    labels, so it shows the separation a classifier actually uses.

    2 groups -> 1 discriminant axis, drawn as a jittered strip plot.
    >=3 groups -> 2 discriminant axes (or as many as available), scatter plot.

    A 5-fold cross-validated balanced accuracy (not resubstitution, which
    would overfit) is reported in the title so the figure is honest about
    how much separation is real vs an artifact of the projection itself.

    Returns (Z, cv_balanced_accuracy).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    P = get_palette(); apply_style()
    cats = list(group_sizes.keys())
    n_cats = len(cats)
    y = np.concatenate([[i] * n for i, n in enumerate(group_sizes.values())])
    n_components = min(n_cats - 1, 2)

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    Z = lda.fit_transform(X, y)

    cv = StratifiedKFold(n_splits=min(cv_splits, min(group_sizes.values())), shuffle=True, random_state=seed)
    cv_acc = float(cross_val_score(
        LinearDiscriminantAnalysis(), X, y, cv=cv, scoring="balanced_accuracy").mean())

    if n_cats <= 2:
        colors_cycle = [P["BLUE"], P["CORAL"]]
    else:
        cmap = plt.get_cmap("tab10" if n_cats <= 10 else "tab20")
        colors_cycle = [cmap(i % cmap.N) for i in range(n_cats)]

    bounds = np.cumsum([0] + list(group_sizes.values()))
    slices = {cat: slice(bounds[i], bounds[i + 1]) for i, cat in enumerate(cats)}

    fig, ax = plt.subplots(figsize=(6, 5.5) if n_components == 2 else (6.5, 4))
    rng = np.random.RandomState(seed)

    if n_components == 1:
        for i, cat in enumerate(cats):
            sl = slices[cat]
            jitter = rng.uniform(-0.35, 0.35, group_sizes[cat])
            ax.scatter(Z[sl, 0], np.full(group_sizes[cat], i) + jitter, c=[colors_cycle[i]],
                       label=f"{cat} (n={group_sizes[cat]})", s=35, alpha=0.7,
                       edgecolors="white", lw=0.35, zorder=3)
        ax.set_yticks(range(n_cats)); ax.set_yticklabels(cats)
        ax.set_xlabel("LD1 (linear discriminant axis)", fontsize=9)
        ax.grid(axis="x", alpha=0.22)
    else:
        for i, cat in enumerate(cats):
            sl = slices[cat]
            _draw_confidence_ellipse(ax, Z[sl], colors_cycle[i], alpha=0.12, lw=1.5,
                                      linestyle="--", zorder=1)
            ax.scatter(Z[sl, 0], Z[sl, 1], c=[colors_cycle[i]],
                       label=f"{cat} (n={group_sizes[cat]})", s=40, alpha=0.72,
                       edgecolors="white", lw=0.35, zorder=3)
        ax.set_xlabel("LD1", fontsize=9)
        ax.set_ylabel("LD2", fontsize=9)
        ax.grid(True, alpha=0.22)

    ax.set_title(f"{title or 'LDA of embeddings'}\n"
                 f"5-fold CV balanced accuracy = {cv_acc:.3f}",
                 fontsize=10, color=P["NAVY"], fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.85)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return Z, cv_acc
