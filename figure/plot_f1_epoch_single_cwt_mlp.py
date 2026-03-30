import argparse
from pathlib import Path

import pandas as pd

from plot_f1_epoch_single import find_model_block, plot_single


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Macro-F1 vs Epoch for CWT-MLP")
    ap.add_argument("--csv", type=str, default="figure/F1-Epoch.csv")
    ap.add_argument("--start_epoch", type=int, default=0)
    ap.add_argument("--smooth", type=str, choices=["ema", "rolling"], default="ema")
    ap.add_argument("--ema_alpha", type=float, default=0.30)
    ap.add_argument("--window", type=int, default=7)
    ap.add_argument("--x_min", type=float, default=20.0)
    ap.add_argument("--x_max", type=float, default=200.0)
    ap.add_argument("--y_min", type=float, default=0.30)
    ap.add_argument("--y_max", type=float, default=0.75)
    ap.add_argument("--out_png", type=str, default="")
    ap.add_argument("--out_svg", type=str, default="")
    args = ap.parse_args()

    model_name = "CWT-MLP"
    csv_path = Path(args.csv).resolve()
    df_all = pd.read_csv(csv_path)
    df_model = find_model_block(df_all, model_name=model_name)

    out_png = Path(args.out_png).resolve() if args.out_png.strip() else csv_path.parent / "macro_f1_epoch_single_cwt_mlp.png"
    out_svg = Path(args.out_svg).resolve() if args.out_svg.strip() else csv_path.parent / "macro_f1_epoch_single_cwt_mlp.svg"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)

    plot_single(
        df_model,
        model_name=model_name,
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

    print(f"[OK] model={model_name}")
    print(f"[OUT] {out_png}")
    print(f"[OUT] {out_svg}")


if __name__ == "__main__":
    main()
