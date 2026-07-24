"""Shared color palette and matplotlib style used by every figure in the package."""

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
    P = get_palette()
    plt.rcParams.update({
        "figure.facecolor": P["WHITE"], "axes.facecolor": P["WHITE"],
        "axes.edgecolor":   P["LGRAY"], "axes.labelcolor": P["NAVY"],
        "xtick.color":      P["GRAY"],  "ytick.color":     P["GRAY"],
        "text.color":       P["NAVY"],  "grid.color":      P["LGRAY"],
        "grid.linewidth":   0.5,        "font.family":     "sans-serif",
        "font.size":        10,
    })
