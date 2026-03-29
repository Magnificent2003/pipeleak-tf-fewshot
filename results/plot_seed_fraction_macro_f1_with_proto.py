import argparse
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


_THIS_DIR = Path(__file__).resolve().parent
_MPL_DIR = _THIS_DIR / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib.pyplot as plt


Z_95 = 1.96
FRACTIONS = [1.0, 0.8, 0.6, 0.4]

MODEL_STYLE = {
    "DarkNet": {"color": "#111111", "marker": "s", "label": "DarkNet-19 (warm-start HCL)"},
    "MFCC-MLP": {"color": "#1f4fff", "marker": "o", "label": "MFCC-MLP-4cls"},
    "CWGF(MFCC+DarkNet)": {"color": "#e31a1c", "marker": "v", "label": "CWGF"},
    "Proto-DarkNet": {
        "color": "#1b9e77",
        "marker": "D",
        "label": "Prototypical Networks (DarkNet encoder)",
    },
}


def load_mfcc_darknet_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"fraction", "model", "mean", "sd", "n"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["sd"] = pd.to_numeric(df["sd"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df = df.dropna(subset=["fraction", "model", "mean", "sd", "n"])
    df = df[df["fraction"].isin(FRACTIONS)]
    df = df[df["model"].isin(["DarkNet", "MFCC-MLP", "CWGF(MFCC+DarkNet)"])]
    return df


def load_proto_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    required = {"fraction", "macro_f1_mean", "macro_f1_sd", "n"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"{path} missing columns: {sorted(miss)}")

    df["fraction"] = pd.to_numeric(df["fraction"], errors="coerce")
    df["macro_f1_mean"] = pd.to_numeric(df["macro_f1_mean"], errors="coerce")
    df["macro_f1_sd"] = pd.to_numeric(df["macro_f1_sd"], errors="coerce")
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df = df.dropna(subset=["fraction", "macro_f1_mean", "macro_f1_sd", "n"])
    df = df[df["fraction"].isin(FRACTIONS)]

    out = pd.DataFrame(
        {
            "fraction": df["fraction"],
            "model": "Proto-DarkNet",
            "mean": df["macro_f1_mean"],
            "sd": df["macro_f1_sd"],
            "n": df["n"],
        }
    )
    return out


def build_curves(df: pd.DataFrame) -> Dict[str, Dict[str, np.ndarray]]:
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for model in MODEL_STYLE.keys():
        sub = df[df["model"] == model].copy()
        by_frac = {float(r["fraction"]): r for _, r in sub.iterrows()}
        mean_arr, sd_arr, n_arr = [], [], []
        for f in FRACTIONS:
            r = by_frac.get(float(f), None)
            if r is None:
                mean_arr.append(np.nan)
                sd_arr.append(np.nan)
                n_arr.append(np.nan)
            else:
                mean_arr.append(float(r["mean"]))
                sd_arr.append(float(r["sd"]))
                n_arr.append(float(r["n"]))

        mean_arr = np.array(mean_arr, dtype=np.float64)
        sd_arr = np.array(sd_arr, dtype=np.float64)
        n_arr = np.array(n_arr, dtype=np.float64)
        ci_arr = Z_95 * (sd_arr / np.sqrt(np.maximum(n_arr, 1.0)))
        curves[model] = {"mean": mean_arr, "sd": sd_arr, "n": n_arr, "ci": ci_arr}
    return curves


def plot_macro_f1_curves(curves: Dict[str, Dict[str, np.ndarray]], out_png: Path, out_svg: Path) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2

    x = np.arange(len(FRACTIONS), dtype=np.float64)
    fig, ax = plt.subplots(figsize=(9.6, 6.8), dpi=300)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")

    for model, style in MODEL_STYLE.items():
        if model not in curves:
            continue
        y = curves[model]["mean"]
        ci = curves[model]["ci"]
        ax.plot(
            x,
            y,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.2,
            markersize=8.0,
            label=style["label"],
            zorder=5,
        )
        ax.fill_between(x, y - ci, y + ci, color=style["color"], alpha=0.12, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{f:.1f}" for f in FRACTIONS], fontsize=13)
    ax.set_xlabel("Train fraction", fontsize=18, labelpad=12)
    ax.set_ylabel("Macro-F1 score", fontsize=18, labelpad=12)
    ax.tick_params(axis="both", which="major", labelsize=13, length=8, width=1.2)
    ax.minorticks_off()
    ax.set_ylim(0.50, 0.75)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.legend(loc="best", frameon=False, fontsize=12)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--summary_mfcc_darknet",
        type=str,
        default=str(
            project_root
            / "runs"
            / "auto_seed_fraction_mfcc_darknet_cwgf_20260322-130620"
            / "csv"
            / "summary_macro_f1_mean_sd.csv"
        ),
    )
    ap.add_argument(
        "--summary_proto",
        type=str,
        default=str(
            project_root
            / "runs"
            / "auto_seed_fraction_proto_darknet4_20260328-000438"
            / "csv"
            / "summary_fraction_proto_darknet4_macro_f1_mean_sd.csv"
        ),
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default=str(
            project_root / "runs" / "auto_seed_fraction_proto_darknet4_20260328-000438" / "csv"
        ),
    )
    args = ap.parse_args()

    p_mfcc = Path(args.summary_mfcc_darknet).resolve()
    p_proto = Path(args.summary_proto).resolve()
    out_dir = Path(args.out_dir).resolve()

    df_main = load_mfcc_darknet_summary(p_mfcc)
    df_proto = load_proto_summary(p_proto)
    df_all = pd.concat([df_main, df_proto], ignore_index=True)

    curves = build_curves(df_all)
    out_png = out_dir / "figure_macro_f1_fraction_4models.png"
    out_svg = out_dir / "figure_macro_f1_fraction_4models.svg"
    plot_macro_f1_curves(curves, out_png=out_png, out_svg=out_svg)

    print(f"[OK] saved: {out_png}")
    print(f"[OK] saved: {out_svg}")


if __name__ == "__main__":
    main()
