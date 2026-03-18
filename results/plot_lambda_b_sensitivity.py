import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


def _split_md_row(line: str) -> List[str]:
    text = line.strip()
    if text.startswith("|"):
        text = text[1:]
    if text.endswith("|"):
        text = text[:-1]
    return [c.strip() for c in text.split("|")]


def _is_alignment_row(cells: Sequence[str]) -> bool:
    pattern = re.compile(r"^:?-{3,}:?$")
    return all(bool(pattern.match(c.replace(" ", ""))) for c in cells)


def parse_markdown_table(table_path: Path) -> List[Dict[str, str]]:
    lines = [ln.strip() for ln in table_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    lines = [ln for ln in lines if ln.startswith("|")]
    if len(lines) < 3:
        raise ValueError(f"markdown table seems invalid: {table_path}")

    header = _split_md_row(lines[0])
    start_idx = 1
    if _is_alignment_row(_split_md_row(lines[1])):
        start_idx = 2

    rows: List[Dict[str, str]] = []
    for ln in lines[start_idx:]:
        cells = _split_md_row(ln)
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def try_float(v: str) -> float:
    return float(v.strip())


def fit_curve_logx(x_vals: Sequence[float], y_vals: Sequence[float], degree: int = 3):
    if len(x_vals) != len(y_vals):
        raise ValueError("x/y length mismatch")
    if len(x_vals) < 2:
        raise ValueError("need at least two points for fitting")

    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    logx = np.log10(x)

    deg = min(int(degree), len(x_vals) - 1)
    coeff = np.polyfit(logx, y, deg=deg)
    poly = np.poly1d(coeff)
    return poly


def configure_times_new_roman(font_path: str = "") -> str:
    # Highest priority: user-provided font file.
    if font_path.strip():
        fp = Path(font_path).resolve()
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            fam = font_manager.FontProperties(fname=str(fp)).get_name()
            plt.rcParams["font.family"] = fam
            return fam

    # Then try system-installed Times New Roman families.
    candidates = [
        "Times New Roman",
        "TimesNewRoman",
        "Times New Roman MT",
        "Times",
    ]
    for fam in candidates:
        try:
            font_manager.findfont(fam, fallback_to_default=False)
            plt.rcParams["font.family"] = fam
            return fam
        except Exception:
            continue

    # Fallback: keep a serif font to avoid repeated findfont warnings.
    plt.rcParams["font.family"] = "DejaVu Serif"
    return "DejaVu Serif"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_md", type=str, required=True, help="Path to markdown table file.")
    ap.add_argument("--lambda_c", type=float, default=None, help="Filter a specific λc value.")
    ap.add_argument("--out_png", type=str, default="")
    ap.add_argument("--out_svg", type=str, default="")
    ap.add_argument(
        "--fit_degree",
        type=int,
        default=5,
        help="Polynomial degree on log10(λb) for fitted dashed curves.",
    )
    ap.add_argument(
        "--font_path",
        type=str,
        default="",
        help="Optional path to a Times New Roman .ttf/.otf file.",
    )
    args = ap.parse_args()

    table_path = Path(args.table_md).resolve()
    if not table_path.exists():
        raise FileNotFoundError(f"missing file: {table_path}")

    rows = parse_markdown_table(table_path)
    if not rows:
        raise ValueError(f"no data rows found in: {table_path}")

    headers = list(rows[0].keys())
    if len(headers) < 3:
        raise ValueError("table must contain at least 3 columns: λc, λb and one metric column")
    if int(args.fit_degree) < 1:
        raise ValueError("--fit_degree must be >= 1")

    lambda_c_col = headers[0]
    lambda_b_col = headers[1]
    model_cols = headers[2:]

    # Convert lambda values
    for r in rows:
        r[lambda_c_col] = try_float(r[lambda_c_col])
        r[lambda_b_col] = try_float(r[lambda_b_col])
        for c in model_cols:
            r[c] = try_float(r[c])

    unique_lc = sorted({r[lambda_c_col] for r in rows})
    if args.lambda_c is not None:
        target_lc = float(args.lambda_c)
    else:
        if len(unique_lc) > 1:
            raise ValueError(
                f"multiple λc values found {unique_lc}. Please set --lambda_c to choose one."
            )
        target_lc = unique_lc[0]

    selected = [r for r in rows if float(r[lambda_c_col]) == target_lc]
    if not selected:
        raise ValueError(f"no rows found for λc={target_lc}")
    selected.sort(key=lambda x: float(x[lambda_b_col]))

    x_vals = [float(r[lambda_b_col]) for r in selected]
    all_y_vals: List[float] = []

    font_used = configure_times_new_roman(args.font_path)
    plt.figure(figsize=(8.4, 5.4))
    ax = plt.gca()

    # Highlight suggested lambda range.
    ax.axvspan(0.2, 0.4, color="#f8c471", alpha=0.22, label="Suggested λ range: 0.2-0.4")
    ax.text(0.215, 0.985, "Suggested region", transform=ax.get_xaxis_transform(),
            fontsize=10, va="top", color="#7d6608")

    # Non-uniform x axis to emphasize the peak area.
    ax.set_xscale("log")
    x_fit = np.logspace(np.log10(0.1), np.log10(2.0), 240)

    for col in model_cols:
        y_vals = [float(r[col]) for r in selected]
        all_y_vals.extend(y_vals)
        poly = fit_curve_logx(x_vals, y_vals, degree=int(args.fit_degree))
        y_fit = poly(np.log10(x_fit))
        # Avoid polynomial tails going out of visible score range.
        y_fit = np.clip(y_fit, 0.0, 1.0)

        # Fitted curve: dashed (no solid line).
        ax.plot(x_fit, y_fit, linestyle="--", linewidth=2.0, label=col)
        # Keep original observations as markers only.
        ax.scatter(x_vals, y_vals, s=28, alpha=0.9)

    if all_y_vals:
        y_min = max(0.0, min(all_y_vals) - 0.02)
        y_max = min(1.0, max(all_y_vals) + 0.02)
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)

    ax.set_xlim(0.1, 2.0)
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "1.0", "2.0"])

    ax.set_xlabel("Lambda b (λb)")
    ax.set_ylabel("F1 Score")
    ax.set_title(f"λb Sensitivity Analysis (λc={target_lc:g})")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False)
    plt.tight_layout()

    if args.out_png.strip():
        out_png = Path(args.out_png).resolve()
    else:
        out_png = table_path.with_name("lambda_b_sensitivity_curve.png")
    if args.out_svg.strip():
        out_svg = Path(args.out_svg).resolve()
    else:
        out_svg = table_path.with_name("lambda_b_sensitivity_curve.svg")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    print(f"[INFO] Font used: {font_used}")
    if font_used.lower() not in {"times new roman", "timesnewroman", "times new roman mt", "times"}:
        print("[WARN] Times New Roman not found. Use --font_path to provide a TTF/OTF file if needed.")
    print(f"[OK] PNG saved: {out_png}")
    print(f"[OK] SVG saved: {out_svg}")


if __name__ == "__main__":
    main()
