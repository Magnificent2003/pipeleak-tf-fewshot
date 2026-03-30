import argparse
import os
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_MPL_DIR = _THIS_DIR / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

import matplotlib.pyplot as plt


METHOD_ORDER = [
    "Baseline (flat 4-class)",
    "Hier-no-Cons",
    "HCL (from start)",
    "HCL (warm-start)",
]

METHOD_STYLE: Dict[str, Dict[str, object]] = {
    "Baseline (flat 4-class)": {"color": "#2B44D2", "lw": 1.5},
    "Hier-no-Cons": {"color": "#D88935", "lw": 1.5},
    "HCL (from start)": {"color": "#111111", "lw": 1.5},
    "HCL (warm-start)": {"color": "#F01111", "lw": 3.0},
}


def strip_suffix(col_name: str) -> str:
    return re.sub(r"\.\d+$", "", col_name)


def find_model_block(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    cols = list(df.columns)
    try:
        i = cols.index(model_name)
    except ValueError as e:
        raise ValueError(f"Model '{model_name}' not found. Available: {cols}") from e

    if i + 4 >= len(cols):
        raise ValueError(f"Model block for '{model_name}' is incomplete in CSV.")

    sub = pd.DataFrame()
    sub["epoch"] = pd.to_numeric(df.iloc[:, i], errors="coerce")
    for j in range(1, 5):
        col = cols[i + j]
        base = strip_suffix(col)
        sub[base] = pd.to_numeric(df.iloc[:, i + j], errors="coerce")

    required = {"epoch"} | set(METHOD_ORDER)
    missing = required - set(sub.columns)
    if missing:
        raise ValueError(f"Missing required columns for '{model_name}': {sorted(missing)}")

    sub = sub.dropna(subset=["epoch"]).copy()
    sub["epoch"] = sub["epoch"].astype(int)
    sub = sub.sort_values("epoch")
    return sub


def smooth_series(y: pd.Series, mode: str, ema_alpha: float, window: int) -> pd.Series:
    if mode == "ema":
        return y.ewm(alpha=ema_alpha, adjust=False).mean()
    if mode == "rolling":
        return y.rolling(window=window, min_periods=1, center=True).mean()
    raise ValueError(f"Unsupported smooth mode: {mode}")


def plot_single(
    df: pd.DataFrame,
    model_name: str,
    start_epoch: int,
    smooth_mode: str,
    ema_alpha: float,
    window: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    out_png: Path,
    out_svg: Path,
) -> None:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 1.8

    x = df["epoch"].to_numpy()
    keep = x >= start_epoch
    x = x[keep]

    fig, ax = plt.subplots(figsize=(10.4, 6.2), dpi=300)
    ax.set_facecolor("#F7F7F7")

    if x_max > 100:
        ax.axvspan(100, x_max, color="#F4DDDD", alpha=0.22, zorder=0)
        ax.axvline(100, color="black", linestyle="--", linewidth=3.2, zorder=3)

    y_min_collect = []
    y_max_collect = []

    for m in METHOD_ORDER:
        y_raw_full = df[m].to_numpy(dtype=float)
        y_raw = y_raw_full[keep]
        y_s = smooth_series(pd.Series(y_raw), mode=smooth_mode, ema_alpha=ema_alpha, window=window).to_numpy()

        st = METHOD_STYLE[m]
        color = st["color"]
        lw = st["lw"]

        # Raw noisy trace as evidence.
        ax.plot(x, y_raw, color=color, linewidth=1.1, alpha=0.20, zorder=1)
        # Smoothed main trace.
        ax.plot(x, y_s, color=color, linewidth=lw, alpha=0.98, zorder=2, label=m)

        y_min_collect.append(np.nanmin(y_s))
        y_max_collect.append(np.nanmax(y_s))

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Epoch", fontsize=19)
    ax.set_ylabel("Validation Macro-F1 Score", fontsize=19)
    ax.tick_params(axis="both", labelsize=15, width=1.6, length=8)
    ax.grid(axis="y", linestyle="--", linewidth=1.0, alpha=0.28)

    if x_max > 100:
        text_x = min(max(103.0, 100 + 0.02 * (x_max - x_min)), x_max - 1.0)
    else:
        text_x = x_min + 0.55 * (x_max - x_min)

    ax.text(
        text_x,
        y_min + 0.5175 * (y_max - y_min),
        "Consistency Regularization Activated",
        fontsize=15.5,
        ha="left",
        va="bottom",
        color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.68, pad=2.0),
    )
    ax.text(text_x, y_min + 0.19 * (y_max - y_min), "Warm-start point", fontsize=16, ha="left", va="bottom", color="black")

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.08, 0.02),
        ncol=1,
        frameon=False,
        fontsize=15,
        handlelength=2.0,
        columnspacing=1.4,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=600)
    fig.savefig(out_svg)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot one-model Macro-F1 vs Epoch from figure/F1-Epoch.csv")
    ap.add_argument("--csv", type=str, default="figure/F1-Epoch.csv")
    ap.add_argument("--model", type=str, default="MFCC-MLP")
    ap.add_argument("--start_epoch", type=int, default=0)
    ap.add_argument("--smooth", type=str, choices=["ema", "rolling"], default="ema")
    ap.add_argument("--ema_alpha", type=float, default=0.30)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--x_min", type=float, default=0.0)
    ap.add_argument("--x_max", type=float, default=200.0)
    ap.add_argument("--y_min", type=float, default=0.50)
    ap.add_argument("--y_max", type=float, default=0.70)
    ap.add_argument("--out_png", type=str, default="")
    ap.add_argument("--out_svg", type=str, default="")
    args = ap.parse_args()

    csv_path = Path(args.csv).resolve()
    df_all = pd.read_csv(csv_path)
    df_model = find_model_block(df_all, model_name=args.model)

    if args.out_png.strip():
        out_png = Path(args.out_png).resolve()
    else:
        out_png = csv_path.parent / f"macro_f1_epoch_single_{args.model.lower().replace('-', '_')}.png"

    if args.out_svg.strip():
        out_svg = Path(args.out_svg).resolve()
    else:
        out_svg = csv_path.parent / f"macro_f1_epoch_single_{args.model.lower().replace('-', '_')}.svg"

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    plot_single(
        df_model,
        model_name=args.model,
        start_epoch=args.start_epoch,
        smooth_mode=args.smooth,
        ema_alpha=args.ema_alpha,
        window=args.window,
        x_min=args.x_min,
        x_max=args.x_max,
        y_min=args.y_min,
        y_max=args.y_max,
        out_png=out_png,
        out_svg=out_svg,
    )

    print(f"[OK] model={args.model}")
    print(f"[OUT] {out_png}")
    print(f"[OUT] {out_svg}")


if __name__ == "__main__":
    main()
