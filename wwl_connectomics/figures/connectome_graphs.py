"""Structural/functional connectome graph layers and glass-brain figures.

plot_structural_layer / plot_functional_layer generalize graphic_aux's old
plot_multilayer (generic 2-color node split) and plot_multilayer_schaefer
(7-network node coloring) into one pair of atomic, single-panel functions —
both variants were previously always drawn side-by-side in one 2-panel figure.
"""

import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from ..style import apply_style, get_palette


def compute_spring_layout(DTI, thr_percentile=40, seed=42, k=0.5):
    """Spring layout from the thresholded DTI graph, shared by both connectome layers."""
    N = DTI.shape[0]
    DTI = DTI.copy(); np.fill_diagonal(DTI, 0)
    G = nx.Graph(); G.add_nodes_from(range(N))
    vals = DTI[DTI > 0]
    thr = np.percentile(vals, thr_percentile) if len(vals) else 0
    for i in range(N):
        for j in range(i + 1, N):
            if DTI[i, j] > thr:
                G.add_edge(i, j, weight=DTI[i, j])
    return nx.spring_layout(G, seed=seed, k=k)


def _thresholded_graph(M, thr_percentile):
    N = M.shape[0]
    M = M.copy(); np.fill_diagonal(M, 0)
    G = nx.Graph(); G.add_nodes_from(range(N))
    vals = M[M > 0]
    thr = np.percentile(vals, thr_percentile) if len(vals) else 0
    for i in range(N):
        for j in range(i + 1, N):
            if M[i, j] > thr:
                G.add_edge(i, j, weight=M[i, j])
    return G


def plot_structural_layer(DTI, pos, node_colors, save_path, region_names=None,
                           title="Structural layer (DTI)", thr_percentile=40,
                           edge_color=None, node_size=460, legend_handles=None):
    """Single-panel structural (DTI) connectome graph — edge width/alpha ∝ weight."""
    P = get_palette(); apply_style()
    edge_color = edge_color or P["BLUE"]
    N = DTI.shape[0]
    G = _thresholded_graph(DTI, thr_percentile)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.set_title(title, fontsize=10, color=edge_color, pad=8)
    ax.set_aspect("equal"); ax.axis("off")

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=w * 6,
                alpha=0.2 + 0.7 * w, zorder=1)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=node_size, edgecolors=P["WHITE"], linewidths=1.5)
    if region_names is not None:
        nx.draw_networkx_labels(G, pos, ax=ax,
            labels={i: region_names[i] for i in range(N)},
            font_size=7.5, font_color=P["WHITE"], font_weight="bold")

    n_e = G.number_of_edges()
    mean_w = np.mean([d["weight"] for _, _, d in G.edges(data=True)]) if n_e else 0
    ax.text(0.02, 0.02, f"edges: {n_e}  |  mean weight: {mean_w:.2f}",
            transform=ax.transAxes, fontsize=7.5, color=P["GRAY"], va="bottom")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
                  framealpha=0.85, edgecolor=P["LGRAY"], facecolor=P["WHITE"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_functional_layer(FC, pos, node_colors, save_path, region_names=None,
                           title="Functional layer (fMRI)", thr_percentile=60,
                           edge_color=None, size_base=180, size_scale=820,
                           legend_handles=None, annotate_hubs=True):
    """Single-panel functional (fMRI) connectome graph — node size ∝ FC strength."""
    P = get_palette(); apply_style()
    edge_color = edge_color or P["TEAL"]
    N = FC.shape[0]
    G = _thresholded_graph(FC, thr_percentile)

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    ax.set_title(title, fontsize=10, color=edge_color, pad=8)
    ax.set_aspect("equal"); ax.axis("off")

    for u, v, d in G.edges(data=True):
        w = d["weight"]
        x0, y0 = pos[u]; x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=w * 4,
                alpha=0.15 + 0.75 * w, zorder=1)

    fc_str = FC.copy()
    np.fill_diagonal(fc_str, 0)
    fc_str = fc_str.sum(axis=1) / (N - 1)
    ptp = np.ptp(fc_str)
    sizes = size_base + size_scale * (fc_str - fc_str.min()) / (ptp + 1e-9)

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                            node_size=sizes, edgecolors=P["WHITE"], linewidths=1.5)
    if region_names is not None:
        nx.draw_networkx_labels(G, pos, ax=ax,
            labels={i: region_names[i] for i in range(N)},
            font_size=7.5, font_color=P["WHITE"], font_weight="bold")
        if annotate_hubs:
            top3 = np.argsort(fc_str)[::-1][:3]
            hub_txt = "Hubs: " + ", ".join(f"{region_names[i]}({fc_str[i]:.2f})" for i in top3)
            ax.text(0.02, 0.02, hub_txt, transform=ax.transAxes,
                    fontsize=7.5, color=P["GRAY"], va="bottom")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="lower right", fontsize=7.5,
                  framealpha=0.85, edgecolor=P["LGRAY"], facecolor=P["WHITE"])

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    return save_path


def plot_glass_brain(shift_vals, FC, save_path, networks, top_pct=20, display_mode="lyrz"):
    """
    Glass-brain plot: node size ∝ shift, edges = FC (top-shift nodes only).
    Node coordinates/labels come from atlas.get_schaefer_coords (nilearn, cached).
    """
    from nilearn.plotting import plot_connectome
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    from .. import atlas as atlas_mod

    P = get_palette(); apply_style()

    coords, labels = atlas_mod.get_schaefer_coords(n_rois=FC.shape[0])
    node_colors = atlas_mod.node_colors_from_labels(labels, networks)

    shift_vals = np.array(shift_vals, dtype=float)
    s_min, s_max = shift_vals.min(), shift_vals.max()
    sizes = 8 + (shift_vals - s_min) / (s_max - s_min + 1e-9) * 152

    # Symmetrize: FC may come in already row-normalized (e.g. zscore_subject),
    # which breaks exact symmetry — nilearn then silently treats the matrix as
    # a DIRECTED graph and draws large FancyArrow arrowheads instead of plain
    # undirected lines (and its arrow width/head_width ignore edge_kwargs).
    # FC edges are conceptually undirected here, so average with the
    # transpose to restore symmetry before handing off to nilearn.
    FC = (FC + FC.T) / 2
    conn = FC.copy(); np.fill_diagonal(conn, 0)
    mask = shift_vals >= np.percentile(shift_vals, 100 - top_pct)

    coords_top = coords[mask]
    sizes_top = sizes[mask]
    colors_top = np.array(node_colors)[mask]
    conn_top = FC[np.ix_(mask, mask)].copy()
    np.fill_diagonal(conn_top, 0)

    # keep only the strongest ~15% of edges among the top-shift nodes — without
    # this plot_connectome draws every pair (up to N*(N-1)/2), unreadable.
    n_top = conn_top.shape[0]
    off_diag = conn_top[~np.eye(n_top, dtype=bool)]
    nonzero = off_diag[off_diag != 0]
    if nonzero.size > 0:
        edge_thr = np.percentile(np.abs(nonzero), 85)
        conn_top[np.abs(conn_top) < edge_thr] = 0
    n_edges_kept = int((conn_top != 0).sum() / 2)
    print(f"  [glass brain] {n_top} nodes, {n_edges_kept} edges kept (threshold = 85th pct)")

    fig, ax = plt.subplots(figsize=(16, 4.5), facecolor=P["WHITE"])
    plot_connectome(
        conn_top, coords_top, node_color=colors_top, node_size=sizes_top,
        edge_threshold=None, edge_vmin=-1, edge_vmax=1,
        edge_kwargs={"linewidth": 0.6}, display_mode=display_mode, axes=ax,
        black_bg=False, alpha=0.12, colorbar=True, annotate=True, title="",
    )
    fig.suptitle(f"Node size ∝ Shift   |   Edges = FC (top-{top_pct}% nodes)",
                 fontsize=11, fontweight="bold", color=P["NAVY"], y=1.02)

    net_handles = atlas_mod.legend_handles(networks)
    size_levels = [0.0, 0.5, 1.0]
    size_labels = [f"{s_min + v * (s_max - s_min):.2f}" for v in size_levels]
    size_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=P["GRAY"], markeredgecolor=P["NAVY"],
               markeredgewidth=0.5, markersize=np.sqrt(8 + v * 152) * 0.75, label=lbl)
        for v, lbl in zip(size_levels, size_labels)
    ]
    leg_nets = fig.legend(handles=net_handles, labels=list(networks.keys()),
        loc="lower left", bbox_to_anchor=(0.01, -0.01), ncol=len(networks),
        fontsize=8.5, framealpha=0.9, edgecolor=P["LGRAY"], facecolor=P["WHITE"],
        borderpad=0.8, columnspacing=0.8, title="Networks", title_fontsize=8)
    fig.legend(handles=size_handles, labels=size_labels,
        loc="lower right", bbox_to_anchor=(0.91, -0.01), ncol=len(size_handles),
        fontsize=8.5, framealpha=0.9, edgecolor=P["LGRAY"], facecolor=P["WHITE"],
        borderpad=0.8, columnspacing=0.8, title="Shift (min · mid · max)", title_fontsize=8)
    fig.add_artist(leg_nets)

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    print(f"  Saved -> {save_path}")
    return save_path
