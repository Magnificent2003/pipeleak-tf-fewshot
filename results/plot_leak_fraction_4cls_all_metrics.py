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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


Z_95 = 1.96

MODEL_ORDER = [
    "DarkNet-19-4cls",
    "DarkNet-19-4cls (Weighted-CE)",
    "MFCC-MLP-4cls",
    "CWGF (DarkNet-19 & MFCC-MLP)",
]

MODEL_STYLE = {
    "DarkNet-19-4cls": {"color": "#111111", "marker": "s", "label": "DarkNet-19"},
    "DarkNet-19-4cls (Weighted-CE)": {"color": "#1f9f88", "marker": "D", "label": "DarkNet-19 (Weighted-CE)"},
    "MFCC-MLP-4cls": {"color": "#1f4fff", "marker": "o", "label": "MFCC-MLP"},
    "CWGF (DarkNet-19 & MFCC-MLP)": {"color": "#e31a1c", "marker": "v", "label": "CWGF"},
}

NON_FUSION_MODELS = [
    "DarkNet-19-4cls",
    "MFCC-MLP-4cls",
]
FUSION_MODEL = "CWGF (DarkNet-19 & MFCC-MLP)"

METRICS = [
    {
        "key": "valve_recall",
        "title": "Valve Recall",
        "higher_better": True,
    },
    {
        "key": "valve_f1",
        "title": "Valve F1",
        "higher_better": True,
    },
    {
        "key": "cross_parent_error_rate",
        "title": "Cross-Parent Error Rate",
        "higher_better": False,
    },
]


def find_latest_summary(project_root: Path) -> Path:
    runs_dir = project_root / "runs"
    cands = sorted(runs_dir.glob("auto_leak_fraction_4cls_*/csv/summary_mean_sd_by_fraction_model.csv"))
    if not cands:
        raise FileNotFoundError(
            "Cannot find runs/auto_leak_fraction_4cls_*/csv/summary_mean_sd_by_fraction_model.csv"
        )
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    req = {"fraction", "model", "n_seeds"}
    for m in METRICS:
        req.add(f"{m['key']}_mean")
        req.add(f"{m['key']}_sd")
    miss = req - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["n_seeds"] = pd.to_numeric(df["n_seeds"], errors="coerce")
    for m in METRICS:
        df[f"{m['key']}_mean"] = pd.to_numeric(df[f"{m['key']}_mean"], errors="coerce")
        df[f"{m['key']}_sd"] = pd.to_numeric(df[f"{m['key']}_sd"], errors="coerce")
    df = df.dropna(subset=["fraction", "model", "n_seeds"])
    return df


def pivot_curves(df: pd.DataFrame, metric_key: str) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    fracs = np.array(sorted(df["fraction"].unique(), reverse=True), dtype=np.float64)
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].copy()
        by_frac = {float(r["fraction"]): r for _, r in sub.iterrows()}
        means, sds, ns = [], [], []
        for f in fracs:
            row = by_frac.get(float(f))
            if row is None:
                means.append(np.nan)
                sds.append(np.nan)
                ns.append(np.nan)
            else:
                means.append(float(row[f"{metric_key}_mean"]))
                sds.append(float(row[f"{metric_key}_sd"]))
                ns.append(float(row["n_seeds"]))
        means = np.array(means, dtype=np.float64)
        sds = np.array(sds, dtype=np.float64)
        ns = np.array(ns, dtype=np.float64)
        se = sds / np.sqrt(np.maximum(ns, 1.0))
        ci = Z_95 * se
        curves[model] = {"mean": means, "sd": sds, "n": ns, "se": se, "ci": ci}
    return fracs, curves


def estimate_gain(curves: Dict[str, Dict[str, np.ndarray]], higher_better: bool) -> Tuple[np.ndarray, np.ndarray]:
    cwgf = curves[FUSION_MODEL]["mean"]
    cwgf_se = curves[FUSION_MODEL]["se"]

    cand_means = np.vstack([curves[m]["mean"] for m in NON_FUSION_MODELS])
    cand_se = np.vstack([curves[m]["se"] for m in NON_FUSION_MODELS])

    if higher_better:
        idx = np.nanargmax(cand_means, axis=0)
    else:
        idx = np.nanargmin(cand_means, axis=0)

    col = np.arange(cand_means.shape[1], dtype=np.int64)
    best = cand_means[idx, col]
    best_se = cand_se[idx, col]

    # Absolute gain (percentage-point style): not relative ratio.
    if higher_better:
        gain = (cwgf - best) * 100.0
    else:
        gain = (best - cwgf) * 100.0

    # Uncertainty of difference: sqrt(se1^2 + se2^2)
    diff_se = np.sqrt(cwgf_se**2 + best_se**2)
    gain_ci = Z_95 * diff_se * 100.0
    return gain, gain_ci


def auto_ylim(y: np.ndarray, ci: np.ndarray, lo_bound: float = 0.0, hi_bound: float = 1.0) -> Tuple[float, float]:
    vmin = np.nanmin(y - ci)
    vmax = np.nanmax(y + ci)
    span = max(vmax - vmin, 1e-6)
    lo = float(lo_bound)
    hi = min(hi_bound, vmax + 0.18 * span)
    if hi - lo < 0.10:
        mid = 0.5 * (hi + lo)
        lo = max(lo_bound, mid - 0.05)
        hi = min(hi_bound, mid + 0.05)
    return float(lo), float(hi)


def auto_ylim_gain(g: np.ndarray, g_ci: np.ndarray) -> Tuple[float, float]:
    lo = np.nanmin(g - g_ci)
    hi = np.nanmax(g + g_ci)
    m = max(abs(lo), abs(hi), 1.0)
    bound = 1.25 * m
    return float(-bound), float(bound)


def plot_all_metrics(df: pd.DataFrame, out_png: Path, out_svg: Path) -> None:
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.2), dpi=300)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    axes = np.array(axes).reshape(-1)

    for i, m in enumerate(METRICS):
        ax1 = axes[i]
        ax1.set_facecolor("none")
        ax2 = ax1.twinx()
        ax2.set_facecolor("none")
        ax1.set_zorder(3)
        ax1.patch.set_alpha(0.0)

        fracs, curves = pivot_curves(df, m["key"])
        x = np.arange(len(fracs), dtype=np.float64)

        gain_mean, gain_ci = estimate_gain(curves, higher_better=bool(m["higher_better"]))
        ax2.bar(
            x,
            gain_mean,
            width=0.40,
            color="#7ec8e3",
            alpha=0.85,
            edgecolor="none",
            zorder=1,
        )
        ax2.errorbar(
            x,
            gain_mean,
            yerr=gain_ci,
            fmt="none",
            ecolor="#1f78b4",
            elinewidth=1.0,
            capsize=2.8,
            capthick=1.0,
            zorder=2,
        )

        for model in MODEL_ORDER:
            style = MODEL_STYLE[model]
            y = curves[model]["mean"]
            ci = curves[model]["ci"]
            ax1.plot(
                x,
                y,
                color=style["color"],
                marker=style["marker"],
                markersize=7.8,
                linewidth=2.0,
                label=style["label"],
                zorder=5,
            )
            ax1.fill_between(x, y - ci, y + ci, color=style["color"], alpha=0.12, zorder=4)

        all_y = np.concatenate([curves[mn]["mean"] for mn in MODEL_ORDER], axis=0)
        all_ci = np.concatenate([curves[mn]["ci"] for mn in MODEL_ORDER], axis=0)
        ylo, yhi = auto_ylim(all_y, all_ci, lo_bound=0.0, hi_bound=1.0)
        glo, ghi = auto_ylim_gain(gain_mean, gain_ci)

        # Right-axis zero baseline for gain reference only.
        ax2.axhline(
            0.0,
            xmin=0.06,
            xmax=0.99,
            color="#5b8fa8",
            linestyle="--",
            linewidth=1.25,
            alpha=0.72,
            zorder=2.4,
        )

        ax1.set_title(m["title"], fontsize=18, pad=8, fontweight="semibold")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"{f:.1f}" for f in fracs], fontsize=14)
        ax1.set_xlabel("Leak Fraction", fontsize=16, labelpad=6, fontweight="semibold")
        ax1.set_ylabel("Score", fontsize=16, labelpad=6, fontweight="semibold")
        ax2.set_ylabel("CWGF Gain vs Best Single Expert (%)", fontsize=14, labelpad=7, fontweight="semibold")
        ax1.set_ylim(ylo, yhi)
        ax2.set_ylim(glo, ghi)
        ax1.minorticks_off()
        ax2.minorticks_off()
        ax1.tick_params(axis="both", which="major", labelsize=13.5, length=7, width=1.2)
        ax2.tick_params(axis="y", which="major", labelsize=12.5, length=6, width=1.2)

        for s in ax1.spines.values():
            s.set_linewidth(1.2)
        for s in ax2.spines.values():
            s.set_linewidth(1.2)

    line_handles = []
    for model in MODEL_ORDER:
        style = MODEL_STYLE[model]
        line_handles.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linewidth=2.0,
                markersize=7.0,
                label=style["label"],
            )
        )
    bar_patch = mpatches.Patch(color="#7ec8e3", alpha=0.85, label="CWGF gain (%)")
    ci_patch = mpatches.Patch(color="#666666", alpha=0.20, label="95% CI")
    handles = line_handles + [bar_patch, ci_patch]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=6,
        frameon=False,
        fontsize=13.2,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    default_csv = find_latest_summary(project_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=str(default_csv))
    ap.add_argument("--out_dir", type=str, default=str(default_csv.parent))
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_summary(csv_path)
    out_png = out_dir / "figure_leak_fraction_4cls_all_metrics.png"
    out_svg = out_dir / "figure_leak_fraction_4cls_all_metrics.svg"
    plot_all_metrics(df, out_png=out_png, out_svg=out_svg)
    print(f"[CSV]  {csv_path}")
    print(f"[SAVE] {out_png}")
    print(f"[SAVE] {out_svg}")


if __name__ == "__main__":
    main()
