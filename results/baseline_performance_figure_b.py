from pathlib import Path

from baseline_performance_plot_common import plot_metric_pair


base_dir = Path(__file__).resolve().parent

plot_metric_pair(
    csv_path=base_dir / "baseline_performance_top5.csv",
    metrics=["macro-f1", "macro-recall"],
    xlabels=["Macro-F1", "Macro-Recall"],
    ylim=(0.45, 0.85),
    output_png=base_dir / "baseline_comparison_figure_2.png",
    output_svg=base_dir / "baseline_comparison_figure_2.svg",
)
