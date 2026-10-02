"""
Atomic, one-figure-per-function plotting library. Every function here draws
exactly one Figure, saves it to `save_path`, closes it, and returns the path
(plus any derived data a caller might want, e.g. a shift dataframe) — no
function produces more than one PNG.

For the classic combined multi-panel layouts these used to always ship in
(e.g. the old 9-panel scenario figure), see wwl_connectomics.reports, which
composes these atomic functions instead of recomputing anything.
"""

from . import (
    classification, connectome_graphs, distance_plots, group_comparison,
    kernel_space, node_shift, regression, transport,
)

from .classification import (
    build_csv_table, plot_accuracy_bars,
    plot_confusion_matrix, plot_fold_accuracy_bars, plot_h_selection,
    plot_lambda_c_per_fold, plot_permutation_null,
)
from .connectome_graphs import (
    compute_spring_layout, plot_functional_layer, plot_glass_brain,
    plot_structural_layer,
)
from .distance_plots import (
    plot_distance_boxplot_by_group, plot_distance_heatmap,
    plot_distance_intra_inter_hist, plot_distance_stats_table,
)
from .group_comparison import plot_group_pca, plot_lda_projection
from .kernel_space import (
    plot_kernel_matrix, plot_kernel_pca, plot_kernel_pca_colored,
    plot_lambda_sensitivity,
)
from .node_shift import (
    plot_embedding_shift_2d, plot_mean_shift_heatmap,
    plot_shift_boxplot_by_network, plot_shift_comparison_subject,
    plot_shift_distribution_by_network, plot_shift_scatter_groups,
    plot_significant_fraction_by_network, plot_top_regions_shift,
)
from .regression import (
    plot_region_importance_bar, plot_regions_scatter_grid,
    plot_regression_scatter, plot_scatter_regression,
)
from .transport import aggregate_to_networks, plot_ground_distance, plot_transport_plan

__all__ = [
    "classification", "connectome_graphs", "distance_plots", "group_comparison",
    "kernel_space", "node_shift", "regression", "transport",
    "build_csv_table", "plot_accuracy_bars",
    "plot_confusion_matrix", "plot_fold_accuracy_bars", "plot_h_selection",
    "plot_lambda_c_per_fold", "plot_permutation_null",
    "compute_spring_layout", "plot_functional_layer", "plot_glass_brain",
    "plot_structural_layer",
    "plot_distance_boxplot_by_group", "plot_distance_heatmap",
    "plot_distance_intra_inter_hist", "plot_distance_stats_table",
    "plot_group_pca", "plot_lda_projection",
    "plot_kernel_matrix", "plot_kernel_pca", "plot_kernel_pca_colored",
    "plot_lambda_sensitivity",
    "plot_embedding_shift_2d", "plot_mean_shift_heatmap",
    "plot_shift_boxplot_by_network", "plot_shift_comparison_subject",
    "plot_shift_distribution_by_network", "plot_shift_scatter_groups",
    "plot_significant_fraction_by_network", "plot_top_regions_shift",
    "plot_region_importance_bar", "plot_regions_scatter_grid",
    "plot_regression_scatter", "plot_scatter_regression",
    "aggregate_to_networks", "plot_ground_distance", "plot_transport_plan",
]
