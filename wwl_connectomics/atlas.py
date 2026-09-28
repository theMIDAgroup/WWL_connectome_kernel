"""Single source of truth for the Schaefer-100 (7 Yeo networks) layout.

This exact dict used to be copy-pasted verbatim in wwl_benchmark.py (twice),
make_scenario_figures.py and WWL_full_pipeline.py — keep it here and import
it everywhere instead.
"""

from functools import lru_cache

from matplotlib.patches import Patch
from nilearn import datasets
from nilearn.plotting import find_parcellation_cut_coords

NETWORKS = {
    "Vis":         (list(range(0,  14)), "#7B1F9C"),
    "SomMot":      (list(range(14, 28)), "#4682B4"),
    "DorsAttn":    (list(range(28, 40)), "#2E7A2E"),
    "SalVentAttn": (list(range(40, 52)), "#C43BFA"),
    "Limbic":      (list(range(52, 60)), "#8BAF3A"),
    "Cont":        (list(range(60, 76)), "#E89020"),
    "Default":     (list(range(76, 100)), "#CE3E50"),
}
NET_NAMES = list(NETWORKS.keys())

NET_COLORS = []
for _, (_rng, _col) in NETWORKS.items():
    NET_COLORS.extend([_col] * len(_rng))
del _rng, _col

# region index -> RSN color, handy for scatter plots colored by network
REGION_COLOR = {ri: col for _rng, col in NETWORKS.values() for ri in _rng}

# Schaefer-100 label prefix -> canonical network key (labels look like
# "7Networks_LH_Vis_1" etc; this maps the third "_"-separated token).
NET_ALIAS = {name: name for name in NET_NAMES}


@lru_cache(maxsize=1)
def get_schaefer_coords(n_rois=100, yeo_networks=7):
    """MNI coordinates + labels for the Schaefer atlas (nilearn fetch, cached).

    Returns (coords, labels) where coords is (n_rois, 3) and labels excludes
    the "Background" entry nilearn prepends.
    """
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks)
    coords = find_parcellation_cut_coords(atlas.maps)
    labels = [l.decode() if isinstance(l, bytes) else l for l in atlas.labels]
    labels = [l for l in labels if l != "Background"]
    return coords, labels


def node_colors_from_labels(labels, networks=NETWORKS):
    """Per-node RSN color from Schaefer label strings (e.g. '7Networks_LH_Vis_1')."""
    return [networks[NET_ALIAS[l.split("_")[2]]][1] for l in labels]


def legend_handles(networks=NETWORKS):
    """Matplotlib Patch handles for a network-color legend."""
    return [Patch(facecolor=col, label=name) for name, (_, col) in networks.items()]
