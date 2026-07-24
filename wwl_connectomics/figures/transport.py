"""Optimal-transport-plan figures (split from graphic_aux's 2-panel plot_transport_plan)."""

import numpy as np

import matplotlib.pyplot as plt

from ..style import apply_style, get_palette


def aggregate_to_networks(M, networks, agg="sum"):
    """Region-level (N, N) matrix -> network-level (n_net, n_net) matrix."""
    region_lists = [list(v[0]) for v in networks.values()]
    nn = len(region_lists)
    out = np.zeros((nn, nn))
    for i, ri in enumerate(region_lists):
        for j, rj in enumerate(region_lists):
            sub = M[np.ix_(ri, rj)]
            out[i, j] = sub.sum() if agg == "sum" else sub.mean()
    return out


def plot_transport_plan(P_star, networks, save_path, net_names=None):
    """Row-normalized network-level optimal transport plan."""
    P = get_palette(); apply_style()
    net_names = net_names or list(networks.keys())
    from matplotlib.colors import LinearSegmentedColormap
    P_net = aggregate_to_networks(P_star, networks, agg="sum")
    P_norm = P_net / (P_net.sum(axis=1, keepdims=True) + 1e-12)

    cmap = LinearSegmentedColormap.from_list("tr", ["white", "#B5D4F4", "#185FA5", "#0D1B3E"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.suptitle("Optimal transport plan  $P^*$", fontsize=11, color=P["NAVY"])
    nn = len(net_names)
    im = ax.imshow(P_norm, cmap=cmap, aspect="auto", vmin=0, vmax=P_norm.max())
    plt.colorbar(im, ax=ax, label="fraction of mass", shrink=0.85)
    ax.set_xticks(range(nn)); ax.set_xticklabels(net_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(nn)); ax.set_yticklabels(net_names, fontsize=9)
    ax.set_xlabel("Subject B", color=P["GRAY"]); ax.set_ylabel("Subject A", color=P["GRAY"])
    th = P_norm.max() * 0.55
    for i in range(nn):
        for j in range(nn):
            v = P_norm[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > th else P["NAVY"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    print(f"  {save_path}")
    return save_path


def plot_ground_distance(M_dist, networks, save_path, net_names=None):
    """Network-level mean Euclidean ground distance underlying the transport plan."""
    P = get_palette(); apply_style()
    net_names = net_names or list(networks.keys())
    from matplotlib.colors import LinearSegmentedColormap
    M_net = aggregate_to_networks(M_dist, networks, agg="mean")

    cmap = LinearSegmentedColormap.from_list("ds", ["white", "#F5C4B3", "#D85A30", "#7A2810"])
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fig.suptitle("Ground distance $M$ (mean Euclidean)", fontsize=11, color=P["NAVY"])
    nn = len(net_names)
    im = ax.imshow(M_net, cmap=cmap, aspect="auto", vmin=0, vmax=M_net.max())
    plt.colorbar(im, ax=ax, label="mean distance", shrink=0.85)
    ax.set_xticks(range(nn)); ax.set_xticklabels(net_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(nn)); ax.set_yticklabels(net_names, fontsize=9)
    ax.set_xlabel("Subject B", color=P["GRAY"]); ax.set_ylabel("Subject A", color=P["GRAY"])
    th = M_net.max() * 0.55
    for i in range(nn):
        for j in range(nn):
            v = M_net[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > th else P["NAVY"])
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=P["WHITE"])
    plt.close(fig)
    print(f"  {save_path}")
    return save_path
