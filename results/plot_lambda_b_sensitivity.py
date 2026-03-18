import argparse
import re
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D


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


def parse_markdown_tables(table_path: Path) -> List[List[Dict[str, str]]]:
    raw = table_path.read_text(encoding="utf-8")
    blocks = [blk.strip() for blk in raw.split("\n\n") if blk.strip()]
    tables: List[List[Dict[str, str]]] = []

    for blk in blocks:
        lines = [ln.strip() for ln in blk.splitlines() if ln.strip()]
        lines = [ln for ln in lines if ln.startswith("|")]
        if len(lines) < 3:
            continue
        header = _split_md_row(lines[0])
        if not _is_alignment_row(_split_md_row(lines[1])):
            continue

        rows: List[Dict[str, str]] = []
        for ln in lines[2:]:
            cells = _split_md_row(ln)
            if len(cells) != len(header):
                continue
            if _is_alignment_row(cells):
                continue
            rows.append(dict(zip(header, cells)))
        if rows:
            tables.append(rows)
    return tables


def pick_table_for_lambda_b(tables: Sequence[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    # Prefer table that has λb but not λcons (pure lambda-b sensitivity table).
    for rows in tables:
        cols = set(rows[0].keys())
        if "λb" in cols and "λcons" not in cols:
            return rows
    # Fallback: any table containing λb.
    for rows in tables:
        cols = set(rows[0].keys())
        if "λb" in cols:
            return rows
    raise ValueError("No table containing column 'λb' found in markdown.")


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


def find_y_for_x(x_vals: Sequence[float], y_vals: Sequence[float], target_x: float) -> float:
    for x, y in zip(x_vals, y_vals):
        if abs(float(x) - float(target_x)) < 1e-9:
            return float(y)
    raise ValueError(f"required x={target_x} not found in table")


def find_intersection_x(x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, x_max: float = 1.0) -> float:
    """Find the crossing x of y_a and y_b on [x.min(), x_max], nearest to x_max."""
    mask = x <= float(x_max)
    xs = x[mask]
    da = (y_a - y_b)[mask]
    if xs.size < 2:
        return float(x_max)

    sign = np.sign(da)
    change_idx = np.where(sign[:-1] * sign[1:] <= 0)[0]
    if change_idx.size == 0:
        return float(x_max)

    i = int(change_idx[-1])  # nearest crossing to x_max
    x0, x1 = float(xs[i]), float(xs[i + 1])
    d0, d1 = float(da[i]), float(da[i + 1])
    if abs(d1 - d0) < 1e-12:
        return x0
    t = -d0 / (d1 - d0)
    t = max(0.0, min(1.0, t))
    return x0 + t * (x1 - x0)


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


def resolve_table_md(table_md_arg: str) -> Path:
    if table_md_arg.strip():
        p = Path(table_md_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"missing file: {p}")
        return p

    cwd = Path.cwd().resolve()
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    direct_candidates = [
        cwd / "f1_curve_table.md",
        project_root / "f1_curve_table.md",
    ]
    for p in direct_candidates:
        if p.exists():
            return p

    # Fallback: choose the newest table under runs/.
    run_roots = [cwd / "runs", project_root / "runs"]
    all_tables: List[Path] = []
    for root in run_roots:
        if root.exists():
            all_tables.extend(root.glob("**/f1_curve_table.md"))

    all_tables = [p.resolve() for p in all_tables if p.exists()]
    if all_tables:
        all_tables.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return all_tables[0]

    raise FileNotFoundError(
        "No f1_curve_table.md found automatically. "
        "Please pass --table_md /path/to/f1_curve_table.md"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table_md",
        type=str,
        default="",
        help="Path to markdown table file. If omitted, script auto-searches f1_curve_table.md.",
    )
    ap.add_argument("--lambda_c", type=float, default=None, help="Filter a specific λc value.")
    ap.add_argument("--out_png", type=str, default="")
    ap.add_argument("--out_svg", type=str, default="")
    ap.add_argument(
        "--fit_degree",
        type=int,
        default=3,
        help="Polynomial degree on log10(λb) for fitted dashed curves.",
    )
    ap.add_argument(
        "--dash_span_ratio",
        type=float,
        default=0.08,
        help="Only keep dashed prediction segments within this ratio of x-span around join point.",
    )
    ap.add_argument(
        "--font_path",
        type=str,
        default="",
        help="Optional path to a Times New Roman .ttf/.otf file.",
    )
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[WARN] ignored unknown args: {unknown}")

    table_path = resolve_table_md(args.table_md)
    print(f"[INFO] table_md: {table_path}")

    tables = parse_markdown_tables(table_path)
    if not tables:
        raise ValueError(f"no markdown table found in: {table_path}")

    rows = pick_table_for_lambda_b(tables)
    if not rows:
        raise ValueError(f"no data rows found in: {table_path}")

    headers = list(rows[0].keys())
    if len(headers) < 3:
        raise ValueError("table must contain at least 3 columns: λc, λb and one metric column")
    if int(args.fit_degree) < 1:
        raise ValueError("--fit_degree must be >= 1")
    if not (0.0 < float(args.dash_span_ratio) <= 1.0):
        raise ValueError("--dash_span_ratio must be in (0, 1].")

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
    plt.figure(figsize=(6.6, 4.8))
    ax = plt.gca()

    # Highlight suggested lambda range.
    ax.axvspan(0.2, 0.4, color="#f8c471", alpha=0.22)
    ax.text(0.215, 0.985, "Suggested region", transform=ax.get_xaxis_transform(),
            fontsize=10, va="top", color="#7d6608")

    # Non-uniform x axis to emphasize the peak area.
    ax.set_xscale("log")
    x_min, x_max = 0.1, 2.0
    x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 320)
    dash_span = max((x_max - x_min) * float(args.dash_span_ratio), 1e-8)
    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = color_cycle.by_key().get("color", []) if color_cycle is not None else []
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    legend_handles: List[Line2D] = []
    for i, col in enumerate(model_cols):
        color = colors[i % len(colors)]
        y_vals = [float(r[col]) for r in selected]
        all_y_vals.extend(y_vals)

        # 1) Fit only first four points (λ=0.1,0.2,0.3,0.5).
        fit_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x <= 0.5]
        fit_pairs = fit_pairs[:4]
        if len(fit_pairs) < 2:
            fit_pairs = list(zip(x_vals[:4], y_vals[:4]))
        fit_x = [p[0] for p in fit_pairs]
        fit_y = [p[1] for p in fit_pairs]

        poly = fit_curve_logx(fit_x, fit_y, degree=int(args.fit_degree))
        y_fit = poly(np.log10(x_fit))
        # Avoid polynomial tails going out of visible score range.
        y_fit = np.clip(y_fit, 0.0, 1.0)

        # 2) Connect λ=1.0 and 2.0 with solid line, and extend with dashed line.
        y1 = find_y_for_x(x_vals, y_vals, 1.0)
        y2 = find_y_for_x(x_vals, y_vals, 2.0)
        slope = (y2 - y1) / (2.0 - 1.0)
        y_line = y1 + slope * (x_fit - 1.0)
        x_join = find_intersection_x(x_fit, y_fit, y_line, x_max=1.0)
        y_join = y1 + slope * (x_join - 1.0)

        if x_min < x_join:
            x_ext_left = np.linspace(x_min, x_join, 120)
            y_ext_left = y1 + slope * (x_ext_left - 1.0)
            ax.plot(x_ext_left, y_ext_left, linestyle="--", linewidth=1.6, alpha=0.9, color=color)
        ax.plot([x_join, 2.0], [y_join, y2], linestyle="-", linewidth=2.2, color=color)

        # 3) Fitted curve under the straight line is dashed; above stays solid.
        below_mask = y_fit < y_line
        dash_l = max(x_min, x_join - dash_span)
        dash_r = min(x_max, x_join + dash_span)
        dashed_mask = below_mask & (x_fit >= dash_l) & (x_fit <= dash_r)
        y_fit_solid = np.where(~below_mask, y_fit, np.nan)
        y_fit_dashed = np.where(dashed_mask, y_fit, np.nan)
        ax.plot(x_fit, y_fit_solid, linestyle="-", linewidth=2.2, color=color)
        ax.plot(x_fit, y_fit_dashed, linestyle="--", linewidth=2.2, color=color)

        # Keep raw observations as markers.
        ax.scatter(x_vals, y_vals, s=28, alpha=0.9, color=color)

        # 4) Keep legend entries unchanged, but always show solid style.
        legend_handles.append(Line2D([0], [0], color=color, linestyle="-", linewidth=2.2, label=col))

    if all_y_vals:
        y_min = max(0.0, min(all_y_vals) - 0.02)
        y_max = min(1.0, max(all_y_vals) + 0.02)
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)

    ax.set_xlim(x_min, x_max)
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "1.0", "2.0"])

    ax.set_xlabel("λb")
    ax.set_ylabel("Macro-F1 Score")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
    )
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
    plt.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    print(f"[INFO] Font used: {font_used}")
    if font_used.lower() not in {"times new roman", "timesnewroman", "times new roman mt", "times"}:
        print("[WARN] Times New Roman not found. Use --font_path to provide a TTF/OTF file if needed.")
    print(f"[OK] PNG saved: {out_png}")
    print(f"[OK] SVG saved: {out_svg}")


if __name__ == "__main__":
    main()
