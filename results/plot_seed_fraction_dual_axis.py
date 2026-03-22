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


MODEL_ORDER = ["DarkNet", "MFCC-MLP", "CWGF(MFCC+DarkNet)"]
MODEL_STYLE = {
    "DarkNet": {"color": "#111111", "marker": "s", "label": "DarkNet"},
    "MFCC-MLP": {"color": "#1f4fff", "marker": "o", "label": "MFCC-MLP"},
    "CWGF(MFCC+DarkNet)": {"color": "#e31a1c", "marker": "v", "label": "CWGF"},
}


def find_latest_seed_fraction_csv_dir(project_root: Path) -> Path:
    runs_dir = project_root / "runs"
    cands = sorted(runs_dir.glob("auto_seed_fraction_mfcc_darknet_cwgf_*/csv"))
    cands = [p for p in cands if (p / "summary_macro_f1_mean_sd.csv").exists()]
    if not cands:
        raise FileNotFoundError("Cannot find runs/auto_seed_fraction_mfcc_darknet_cwgf_*/csv with summary files.")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def load_long_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"fraction", "model", "mean", "sd", "n"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing required columns: {sorted(miss)}")
    df = df.copy()
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["sd"] = pd.to_numeric(df["sd"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df = df.dropna(subset=["fraction", "model", "mean", "sd", "n"])
    return df


def pivot_model_curves(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    fractions = np.array(sorted(df["fraction"].unique(), reverse=True), dtype=np.float64)
    curves: Dict[str, Dict[str, np.ndarray]] = {}

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].copy()
        by_frac = {float(r["fraction"]): r for _, r in sub.iterrows()}

        means, sds, ns = [], [], []
        for frac in fractions:
            row = by_frac.get(float(frac), None)
            if row is None:
                means.append(np.nan)
                sds.append(np.nan)
                ns.append(np.nan)
            else:
                means.append(float(row["mean"]))
                sds.append(float(row["sd"]))
                ns.append(float(row["n"]))

        means = np.array(means, dtype=np.float64)
        sds = np.array(sds, dtype=np.float64)
        ns = np.array(ns, dtype=np.float64)
        se = sds / np.sqrt(np.maximum(ns, 1.0))
        ci = Z_95 * se
        curves[model] = {"mean": means, "sd": sds, "n": ns, "ci": ci}

    return fractions, curves


def estimate_gain_from_raw(
    raw_csv: Path,
    fractions: np.ndarray,
    metric_cols: Dict[str, str],
) -> Tuple[np.ndarray, np.ndarray]:
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)

    df = pd.read_csv(raw_csv).copy()
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")]
    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    for col in metric_cols.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")

    gain_mean, gain_ci = [], []
    for frac in fractions:
        sub = df[df["fraction"] == float(frac)].copy()
        if sub.empty:
            gain_mean.append(np.nan)
            gain_ci.append(np.nan)
            continue

        best = np.maximum(sub[metric_cols["DarkNet"]].to_numpy(), sub[metric_cols["MFCC-MLP"]].to_numpy())
        cwgf = sub[metric_cols["CWGF(MFCC+DarkNet)"]].to_numpy()
        valid = np.isfinite(best) & np.isfinite(cwgf) & (best > 0)
        vals = (cwgf[valid] - best[valid]) / best[valid] * 100.0

        if vals.size == 0:
            gain_mean.append(np.nan)
            gain_ci.append(np.nan)
            continue

        m = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        ci = Z_95 * (sd / np.sqrt(max(vals.size, 1)))
        gain_mean.append(m)
        gain_ci.append(ci)

    return np.array(gain_mean, dtype=np.float64), np.array(gain_ci, dtype=np.float64)


def estimate_gain_from_aggregated(
    curves: Dict[str, Dict[str, np.ndarray]]
) -> Tuple[np.ndarray, np.ndarray]:
    dark = curves["DarkNet"]["mean"]
    mfcc = curves["MFCC-MLP"]["mean"]
    cwgf = curves["CWGF(MFCC+DarkNet)"]["mean"]

    dark_se = curves["DarkNet"]["sd"] / np.sqrt(np.maximum(curves["DarkNet"]["n"], 1.0))
    mfcc_se = curves["MFCC-MLP"]["sd"] / np.sqrt(np.maximum(curves["MFCC-MLP"]["n"], 1.0))
    cwgf_se = curves["CWGF(MFCC+DarkNet)"]["sd"] / np.sqrt(np.maximum(curves["CWGF(MFCC+DarkNet)"]["n"], 1.0))

    best_is_dark = dark >= mfcc
    best = np.where(best_is_dark, dark, mfcc)
    best_se = np.where(best_is_dark, dark_se, mfcc_se)

    gain = (cwgf - best) / np.maximum(best, 1e-12) * 100.0
    # Delta approximation for ratio uncertainty.
    ratio_se = np.sqrt((cwgf_se / np.maximum(best, 1e-12)) ** 2 + ((cwgf * best_se) / np.maximum(best, 1e-12) ** 2) ** 2)
    gain_ci = Z_95 * ratio_se * 100.0
    return gain, gain_ci


def plot_dual_axis(
    fractions: np.ndarray,
    curves: Dict[str, Dict[str, np.ndarray]],
    gain_mean: np.ndarray,
    gain_ci: np.ndarray,
    metric_label: str,
    left_ylim: Tuple[float, float],
    right_ylim: Tuple[float, float],
    out_png: Path,
    out_svg: Path,
) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["xtick.minor.width"] = 1.0
    plt.rcParams["ytick.minor.width"] = 1.0

    x = np.arange(len(fractions), dtype=np.float64)

    fig, ax1 = plt.subplots(figsize=(10.5, 7.6), dpi=300)
    bg = "#f2f2f2"
    fig.patch.set_facecolor(bg)
    ax1.set_facecolor(bg)

    ax2 = ax1.twinx()
    ax1.set_zorder(3)
    ax1.patch.set_alpha(0.0)

    bars = ax2.bar(
        x,
        gain_mean,
        width=0.40,
        color="#7ec8e3",
        alpha=0.85,
        edgecolor="none",
        zorder=1,
        label="CWGF gain vs best single expert",
    )
    ax2.errorbar(
        x,
        gain_mean,
        yerr=gain_ci,
        fmt="none",
        ecolor="#1f78b4",
        elinewidth=1.1,
        capsize=3,
        capthick=1.1,
        zorder=2,
    )

    line_handles = []
    for model in MODEL_ORDER:
        if model not in curves:
            continue
        style = MODEL_STYLE[model]
        y = curves[model]["mean"]
        ci = curves[model]["ci"]

        h, = ax1.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            markersize=8.5,
            linewidth=2.2,
            label=style["label"],
            zorder=5,
        )
        ax1.fill_between(x, y - ci, y + ci, color=style["color"], alpha=0.12, zorder=4)
        line_handles.append(h)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{f:.1f}" for f in fractions], fontsize=13)
    ax1.set_xlabel("Train fraction", fontsize=18, labelpad=14)
    ax1.set_ylabel(f"{metric_label} score", fontsize=19, labelpad=14)
    ax2.set_ylabel("Gain over best single expert (%)", fontsize=19, labelpad=14)

    ax1.set_ylim(float(left_ylim[0]), float(left_ylim[1]))
    ax2.set_ylim(float(right_ylim[0]), float(right_ylim[1]))

    # Keep only major ticks.
    ax1.minorticks_off()
    ax2.minorticks_off()
    ax1.tick_params(axis="both", which="major", labelsize=13, length=9, width=1.2)
    ax2.tick_params(axis="y", which="major", labelsize=13, length=9, width=1.2)

    bar_patch = mpatches.Patch(color="#7ec8e3", alpha=0.85, label="CWGF gain (%)")
    ci_patch = mpatches.Patch(color="#666666", alpha=0.20, label="95% CI")
    legend_handles = line_handles + [bar_patch, ci_patch]
    ax1.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=False,
        fontsize=11.5,
        handlelength=2.5,
        handletextpad=0.4,
        borderaxespad=0.4,
    )

    for s in ax1.spines.values():
        s.set_linewidth(1.2)
    for s in ax2.spines.values():
        s.set_linewidth(1.2)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def make_one_figure(
    summary_csv: Path,
    raw_csv: Path,
    metric_label: str,
    metric_cols_in_raw: Dict[str, str],
    left_ylim: Tuple[float, float],
    right_ylim: Tuple[float, float],
    out_png: Path,
    out_svg: Path,
) -> None:
    df = load_long_summary(summary_csv)
    fractions, curves = pivot_model_curves(df)

    if raw_csv.exists():
        gain_mean, gain_ci = estimate_gain_from_raw(raw_csv, fractions, metric_cols_in_raw)
    else:
        gain_mean, gain_ci = estimate_gain_from_aggregated(curves)

    plot_dual_axis(
        fractions=fractions,
        curves=curves,
        gain_mean=gain_mean,
        gain_ci=gain_ci,
        metric_label=metric_label,
        left_ylim=left_ylim,
        right_ylim=right_ylim,
        out_png=out_png,
        out_svg=out_svg,
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    latest_csv_dir = find_latest_seed_fraction_csv_dir(project_root)

    ap = argparse.ArgumentParser()
    ap.add_argument("--macro_csv", type=str, default=str(latest_csv_dir / "summary_macro_f1_mean_sd.csv"))
    ap.add_argument("--parent_csv", type=str, default=str(latest_csv_dir / "summary_parent_f1_mean_sd.csv"))
    ap.add_argument("--raw_csv", type=str, default=str(latest_csv_dir / "summary_seed_fraction_metrics.csv"))
    ap.add_argument("--out_dir", type=str, default=str(latest_csv_dir))
    args = ap.parse_args()

    macro_csv = Path(args.macro_csv).resolve()
    parent_csv = Path(args.parent_csv).resolve()
    raw_csv = Path(args.raw_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    make_one_figure(
        summary_csv=macro_csv,
        raw_csv=raw_csv,
        metric_label="Macro-F1",
        metric_cols_in_raw={
            "MFCC-MLP": "mfcc_mlp_4cls_macro_f1",
            "DarkNet": "darknet_4cls_macro_f1",
            "CWGF(MFCC+DarkNet)": "cwgf_mfcc_darknet_4cls_macro_f1",
        },
        left_ylim=(0.40, 0.75),
        right_ylim=(0.0, 8.0),
        out_png=out_dir / "figure7_macro_f1_fraction_dual_axis.png",
        out_svg=out_dir / "figure7_macro_f1_fraction_dual_axis.svg",
    )

    make_one_figure(
        summary_csv=parent_csv,
        raw_csv=raw_csv,
        metric_label="Parent-F1",
        metric_cols_in_raw={
            "MFCC-MLP": "mfcc_mlp_2cls_f1",
            "DarkNet": "darknet_2cls_f1",
            "CWGF(MFCC+DarkNet)": "cwgf_mfcc_darknet_2cls_f1",
        },
        left_ylim=(0.70, 0.925),
        right_ylim=(0.0, 5.0),
        out_png=out_dir / "figure7_parent_f1_fraction_dual_axis.png",
        out_svg=out_dir / "figure7_parent_f1_fraction_dual_axis.svg",
    )

    print(f"[SAVE] {out_dir / 'figure7_macro_f1_fraction_dual_axis.png'}")
    print(f"[SAVE] {out_dir / 'figure7_macro_f1_fraction_dual_axis.svg'}")
    print(f"[SAVE] {out_dir / 'figure7_parent_f1_fraction_dual_axis.png'}")
    print(f"[SAVE] {out_dir / 'figure7_parent_f1_fraction_dual_axis.svg'}")


if __name__ == "__main__":
    main()
