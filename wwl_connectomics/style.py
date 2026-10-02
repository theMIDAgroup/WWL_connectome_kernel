"""Shared color palette and matplotlib style used by every figure in the package."""

import numpy as np
import matplotlib.pyplot as plt


def get_palette():
    return dict(
        NAVY   = "#0D1B3E",
        BLUE   = "#185FA5",
        LBLUE  = "#B5D4F4",
        CORAL  = "#D85A30",
        LCORAL = "#F5C4B3",
        VIOLET = "#534AB7",
        AMBER  = "#BA7517",
        TEAL   = "#0D9488",
        GREEN  = "#3B6D11",
        LGRAY  = "#E8ECF4",
        GRAY   = "#8292A8",
        WHITE  = "#FFFFFF",
    )


def apply_style():
    """Color/grid style only — font is left at matplotlib's own default
    (rcParamsDefault, currently DejaVu Sans) rather than pinned here."""
    P = get_palette()
    plt.rcParams.update({
        "figure.facecolor": P["WHITE"], "axes.facecolor": P["WHITE"],
        "axes.edgecolor":   P["LGRAY"], "axes.labelcolor": "black",
        "xtick.color":      "black",    "ytick.color":     "black",
        "text.color":       "black",     "grid.color":      P["LGRAY"],
        "axes.titlecolor":  "black",
        "grid.linewidth":   0.5,        "font.size":       10,
    })


def draw_violin(ax, data, position, width, color, median_color, alpha=0.65):
    """One colored violin at a single x position (replaces the project's old per-RSN
    boxplot styling). Falls back to a boxplot when there's too little data for a
    kernel-density estimate (violinplot needs >=2 points with nonzero spread)."""
    data = np.asarray(data, dtype=float)
    if len(data) < 2 or np.ptp(data) == 0:
        return ax.boxplot(data, positions=[position], widths=width, patch_artist=True,
                           boxprops=dict(facecolor=color, alpha=alpha),
                           medianprops=dict(color=median_color, lw=2),
                           whiskerprops=dict(color=color), capprops=dict(color=color),
                           showfliers=False)
    parts = ax.violinplot(data, positions=[position], widths=width,
                           showmeans=False, showmedians=True, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_edgecolor(color)
        pc.set_alpha(alpha)
    parts["cmedians"].set_color(median_color)
    parts["cmedians"].set_linewidth(2)
    for key in ("cbars", "cmins", "cmaxes"):
        parts[key].set_color(color)
        parts[key].set_linewidth(1)
    return parts
