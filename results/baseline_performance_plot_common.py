from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from matplotlib import font_manager


BASE_COLORS = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#8172B2",
    "#DD585C",
    "#CCB974",
    "#64B5CD",
    "#E17C05",
    "#8C8C8C",
    "#2B5F37",
]

MODEL_COLOR_OVERRIDES = {
    # Ensure these two are clearly distinct.
    "STFT-ResNet-18": "#6CE9FF",
    "HHT-MLP": "#D40000",
}


def _set_preferred_font():
    # Try hard to use Times New Roman if available on this machine.
    preferred = "Times New Roman"
    base_dir = Path(__file__).resolve().parent
    candidates = [
        str(base_dir / "fonts" / "Times_New_Roman.ttf"),
        str(base_dir / "fonts" / "times.ttf"),
        str(base_dir / "fonts" / "timesnewroman.ttf"),
        "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/times.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/timesnewroman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    existing = [p for p in candidates if Path(p).exists()]
    if existing:
        font_manager.fontManager.addfont(existing[0])
        plt.rcParams["font.family"] = preferred
    else:
        plt.rcParams["font.family"] = preferred
        # Fallback serif list for environments where Times New Roman is absent.
        plt.rcParams["font.serif"] = [preferred, "STIXGeneral", "DejaVu Serif"]


def _load_top5_csv(csv_path: Path):
    df = pd.read_csv(csv_path)
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["sd"] = pd.to_numeric(df["sd"], errors="coerce").fillna(0.0)
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")
    return df.dropna(subset=["metric", "model", "mean", "rank"]).copy()


def _build_palette(df: pd.DataFrame):
    models = sorted(df["model"].dropna().unique().tolist())
    palette = {m: BASE_COLORS[i % len(BASE_COLORS)] for i, m in enumerate(models)}
    palette.update({k: v for k, v in MODEL_COLOR_OVERRIDES.items() if k in palette})
    return palette


def plot_metric_pair(
    csv_path: Path,
    metrics,
    xlabels,
    ylim,
    output_png: Path,
    output_svg: Path,
    figsize=(6, 3),
):
    df = _load_top5_csv(csv_path)
    subsets = [df[df["metric"] == m].sort_values("rank") for m in metrics]

    n_per_group = [len(s) for s in subsets]
    width = 0.8
    x_positions = []
    current_x = 0
    for n in n_per_group:
        xs = list(range(current_x, current_x + n))
        x_positions.extend(xs)
        current_x += n + 1

    scores = []
    errs = []
    models = []
    for s in subsets:
        scores.extend(s["mean"].tolist())
        errs.extend(s["sd"].tolist())
        models.extend(s["model"].tolist())

    _set_preferred_font()
    fig, ax = plt.subplots(figsize=figsize)

    palette = _build_palette(df)
    bar_colors = [palette[m] for m in models]

    bars = ax.bar(
        x_positions,
        scores,
        yerr=errs,
        capsize=3,
        width=width,
        color=bar_colors,
        edgecolor="none",
        error_kw={"elinewidth": 0.8, "ecolor": "#333333", "alpha": 0.85},
    )

    ax.set_ylim(*ylim)
    ax.set_ylabel("Score (%)", fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}"))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)

    centers = []
    start = 0
    for n in n_per_group:
        centers.append(start + (n - 1) / 2 if n > 0 else start)
        start += n + 1
    ax.set_xticks(centers)
    ax.set_xticklabels(xlabels, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.set_title("")
    ax.set_xlabel("")

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.002,
            f"{h*100:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    model_list = list(dict.fromkeys(models))
    handles = [mpatches.Patch(color=palette[m], label=m) for m in model_list]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=9,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_png, bbox_inches="tight", dpi=600, transparent=True)
    plt.savefig(output_svg, bbox_inches="tight")
    plt.show()
