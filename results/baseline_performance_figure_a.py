from pathlib import Path

from baseline_performance_plot_common import plot_metric_pair


base_dir = Path(__file__).resolve().parent

plot_metric_pair(
    csv_path=base_dir / "baseline_performance_top5.csv",
    metrics=["binary-f1", "binary-recall"],
    xlabels=["Binary-F1", "Binary-Recall"],
    ylim=(0.70, 0.95),
    output_png=base_dir / "baseline_comparison_figure_1.png",
    output_svg=base_dir / "baseline_comparison_figure_1.svg",
)
