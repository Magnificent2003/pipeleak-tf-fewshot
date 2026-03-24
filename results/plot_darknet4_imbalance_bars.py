import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_MPL_DIR = _THIS_DIR / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib.pyplot as plt


MODEL_FULLNAME = {
    "DarkNet-19 + CE": "DarkNet-19 +\nStandard Cross-Entropy",
    "DarkNet-19 + Weighted-CE": "DarkNet-19 +\nClass-Weighted Cross-Entropy",
    "DarkNet-19 + Focal": "DarkNet-19 +\nFocal Loss",
}

MODEL_ORDER = [
    "DarkNet-19 + CE",
    "DarkNet-19 + Weighted-CE",
    "DarkNet-19 + Focal",
]

LEFT_METRICS: List[Tuple[str, str, str]] = [
    ("valve_recall", "Valve Recall", "#4C72B0"),
    ("valve_f1", "Valve F1", "#DD8452"),
]
RIGHT_METRIC: Tuple[str, str, str] = ("cross_parent_error_rate", "Cross-Parent Error Rate", "#55A868")


def _find_latest_summary(project_root: Path) -> Path:
    runs = project_root / "runs"
    cands = sorted(runs.glob("auto_darknet4_imbalance_10seeds_*/csv/summary_mean_sd.csv"))
    if not cands:
        raise FileNotFoundError("Cannot find runs/auto_darknet4_imbalance_10seeds_*/csv/summary_mean_sd.csv")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    req = {"model"}
    for metric, _, _ in LEFT_METRICS + [RIGHT_METRIC]:
        req.add(f"{metric}_mean")
        req.add(f"{metric}_sd")
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns in {path}: {sorted(miss)}")

    for metric, _, _ in LEFT_METRICS + [RIGHT_METRIC]:
        df[f"{metric}_mean"] = pd.to_numeric(df[f"{metric}_mean"], errors="coerce")
        df[f"{metric}_sd"] = pd.to_numeric(df[f"{metric}_sd"], errors="coerce")
    df = df.dropna(subset=["model"])
    return df


def reorder_models(df: pd.DataFrame) -> pd.DataFrame:
    idx_map: Dict[str, int] = {k: i for i, k in enumerate(MODEL_ORDER)}

    def _key(name: str) -> int:
        return idx_map.get(name, 999)

    df = df.copy()
    df["_ord"] = df["model"].astype(str).map(_key)
    df = df.sort_values("_ord").drop(columns=["_ord"])
    return df


def _annotate_bars(ax, bars, values: np.ndarray, dy: float, fontsize: float = 11.0) -> None:
    for rect, v in zip(bars, values.tolist()):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + dy,
            f"{float(v):.3f}",
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight="semibold",
        )


def plot_grouped_bars(
    df: pd.DataFrame,
    out_png: Path,
    out_svg: Path,
    left_ylim: Tuple[float, float] = (0.25, 0.65),
    right_ylim: Tuple[float, float] = (0.0, 0.30),
) -> None:
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False

    x = np.arange(len(df), dtype=np.float64)
    width = 0.22
    offsets = np.array([-width, 0.0, width], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12.2, 7.3), dpi=300)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    ax2 = ax.twinx()
    ax2.set_facecolor("none")
    ax.set_zorder(2)
    ax.patch.set_alpha(0.0)

    left_handles = []
    for i, (metric, label, color) in enumerate(LEFT_METRICS):
        y = df[f"{metric}_mean"].to_numpy(dtype=np.float64)
        e = df[f"{metric}_sd"].to_numpy(dtype=np.float64)
        bars = ax.bar(
            x + offsets[i],
            y,
            width=width * 0.92,
            color=color,
            alpha=0.92,
            edgecolor="none",
            linewidth=0.0,
            label=label,
            zorder=3,
        )
        ax.errorbar(
            x + offsets[i],
            y,
            yerr=e,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            capthick=1.0,
            zorder=4,
        )
        _annotate_bars(ax, bars, y, dy=0.008, fontsize=15)
        left_handles.append(bars)

    metric_r, label_r, color_r = RIGHT_METRIC
    y_r = df[f"{metric_r}_mean"].to_numpy(dtype=np.float64)
    e_r = df[f"{metric_r}_sd"].to_numpy(dtype=np.float64)
    bars_r = ax2.bar(
        x + offsets[2],
        y_r,
        width=width * 0.92,
        color=color_r,
        alpha=0.92,
        edgecolor="none",
        linewidth=0.0,
        label=label_r,
        zorder=3,
    )
    ax2.errorbar(
        x + offsets[2],
        y_r,
        yerr=e_r,
        fmt="none",
        ecolor="black",
        elinewidth=1.0,
        capsize=3,
        capthick=1.0,
        zorder=4,
    )
    _annotate_bars(ax2, bars_r, y_r, dy=0.007, fontsize=15)

    labels = [MODEL_FULLNAME.get(m, m) for m in df["model"].astype(str).tolist()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=18, fontweight="bold")
    ax.set_ylabel("Valve Metrics (Recall / F1)", fontsize=22, fontweight="bold")
    ax2.set_ylabel("Cross-Parent Error Rate", fontsize=22, fontweight="bold")
    ax.set_ylim(float(left_ylim[0]), float(left_ylim[1]))
    ax2.set_ylim(float(right_ylim[0]), float(right_ylim[1]))
    ax.tick_params(axis="y", labelsize=18, length=7, width=1.2)
    ax2.tick_params(axis="y", labelsize=18, length=7, width=1.2)
    ax.tick_params(axis="x", labelsize=17, length=0, pad=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.85, alpha=0.30, zorder=1)
    ax2.grid(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        ncol=1,
        frameon=False,
        fontsize=17,
    )

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_csv = _find_latest_summary(project_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(default_csv))
    ap.add_argument("--out_dir", type=str, default=str(default_csv.parent))
    ap.add_argument("--left_ylim", type=str, default="0.25,0.65")
    ap.add_argument("--right_ylim", type=str, default="0.0,0.3")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(csv_path)
    df = reorder_models(df)

    l_lo, l_hi = [float(x.strip()) for x in args.left_ylim.split(",")]
    r_lo, r_hi = [float(x.strip()) for x in args.right_ylim.split(",")]

    out_png = out_dir / "darknet4_imbalance_grouped_bars_dual_axis.png"
    out_svg = out_dir / "darknet4_imbalance_grouped_bars_dual_axis.svg"
    plot_grouped_bars(df, out_png=out_png, out_svg=out_svg, left_ylim=(l_lo, l_hi), right_ylim=(r_lo, r_hi))

    print(f"[CSV]  {csv_path}")
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_svg}")


if __name__ == "__main__":
    main()
