import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_THIS_DIR = Path(__file__).resolve().parent
_MPL_DIR = _THIS_DIR / ".mplconfig"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

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
    for rows in tables:
        cols = set(rows[0].keys())
        if "λb" in cols and "λcons" not in cols:
            return rows
    for rows in tables:
        cols = set(rows[0].keys())
        if "λb" in cols:
            return rows
    raise ValueError("No table containing column 'λb' found in markdown.")


def pick_table_for_lambda_cons(tables: Sequence[List[Dict[str, str]]]) -> List[Dict[str, str]]:
    for rows in tables:
        cols = set(rows[0].keys())
        if "λcons" in cols:
            return rows
    raise ValueError("No table containing column 'λcons' found in markdown.")


def try_float(v: str) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return float(str(v).strip())


def fit_curve_logx_for_lambda_b(x_vals: Sequence[float], y_vals: Sequence[float], degree: int = 3):
    if len(x_vals) != len(y_vals):
        raise ValueError("x/y length mismatch")
    if len(x_vals) < 2:
        raise ValueError("need at least two points for fitting")
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    logx = np.log10(x)
    deg = min(int(degree), len(x_vals) - 1)
    coeff = np.polyfit(logx, y, deg=deg)
    return np.poly1d(coeff)


def fit_curve_linear_for_lambda_cons(x_vals: Sequence[float], y_vals: Sequence[float], degree: int = 2):
    if len(x_vals) != len(y_vals):
        raise ValueError("x/y length mismatch")
    if len(x_vals) < 2:
        raise ValueError("need at least two points for fitting")
    x = np.asarray(x_vals, dtype=np.float64)
    y = np.asarray(y_vals, dtype=np.float64)
    deg = min(int(degree), len(x_vals) - 1)
    coeff = np.polyfit(x, y, deg=deg)
    return np.poly1d(coeff)


def find_y_for_x(x_vals: Sequence[float], y_vals: Sequence[float], target_x: float) -> float:
    for x, y in zip(x_vals, y_vals):
        if abs(float(x) - float(target_x)) < 1e-9:
            return float(y)
    raise ValueError(f"required x={target_x} not found in table")


def find_intersection_x(x: np.ndarray, y_a: np.ndarray, y_b: np.ndarray, x_max: float = 1.0) -> float:
    mask = x <= float(x_max)
    xs = x[mask]
    da = (y_a - y_b)[mask]
    if xs.size < 2:
        return float(x_max)

    sign = np.sign(da)
    change_idx = np.where(sign[:-1] * sign[1:] <= 0)[0]
    if change_idx.size == 0:
        return float(x_max)

    i = int(change_idx[-1])
    x0, x1 = float(xs[i]), float(xs[i + 1])
    d0, d1 = float(da[i]), float(da[i + 1])
    if abs(d1 - d0) < 1e-12:
        return x0
    t = -d0 / (d1 - d0)
    t = max(0.0, min(1.0, t))
    return x0 + t * (x1 - x0)


def configure_times_new_roman(font_path: str = "") -> str:
    if font_path.strip():
        fp = Path(font_path).resolve()
        if fp.exists():
            font_manager.fontManager.addfont(str(fp))
            fam = font_manager.FontProperties(fname=str(fp)).get_name()
            plt.rcParams["font.family"] = fam
            return fam

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


def preprocess_lambda_b_rows(rows: List[Dict[str, str]], target_lc: float):
    headers = list(rows[0].keys())
    lambda_c_col = headers[0]
    lambda_b_col = headers[1]
    model_cols = headers[2:]

    for r in rows:
        r[lambda_c_col] = try_float(r[lambda_c_col])
        r[lambda_b_col] = try_float(r[lambda_b_col])
        for c in model_cols:
            r[c] = try_float(r[c])

    selected = [r for r in rows if float(r[lambda_c_col]) == target_lc]
    if not selected:
        raise ValueError(f"no rows found for λc={target_lc}")
    selected.sort(key=lambda x: float(x[lambda_b_col]))
    return lambda_b_col, model_cols, selected


def preprocess_lambda_cons_rows(rows: List[Dict[str, str]], target_lc: float, target_lb: float):
    headers = list(rows[0].keys())
    lambda_c_col = headers[0]
    lambda_b_col = headers[1]
    lambda_cons_col = headers[2]
    model_cols = headers[3:]

    for r in rows:
        r[lambda_c_col] = try_float(r[lambda_c_col])
        r[lambda_b_col] = try_float(r[lambda_b_col])
        r[lambda_cons_col] = try_float(r[lambda_cons_col])
        for c in model_cols:
            r[c] = try_float(r[c])

    selected = [
        r
        for r in rows
        if abs(float(r[lambda_c_col]) - target_lc) < 1e-9 and abs(float(r[lambda_b_col]) - target_lb) < 1e-9
    ]
    if not selected:
        raise ValueError(f"no rows found for λc={target_lc}, λb={target_lb}")
    selected.sort(key=lambda x: float(x[lambda_cons_col]))
    return lambda_cons_col, model_cols, selected


def draw_lambda_b_panel(
    ax,
    selected: List[Dict[str, float]],
    lambda_b_col: str,
    model_cols: List[str],
    fit_degree: int,
    dash_span_ratio: float,
) -> List[Line2D]:
    x_vals = [float(r[lambda_b_col]) for r in selected]
    all_y_vals: List[float] = []

    ax.axvspan(0.2, 0.4, color="#f8c471", alpha=0.22)
    ax.text(
        0.215,
        0.985,
        "Recommended region",
        transform=ax.get_xaxis_transform(),
        fontsize=14,  # original 10 -> +4
        va="top",
        color="#7d6608",
    )

    ax.set_xscale("log")
    x_min, x_max = 0.1, 2.0
    x_fit = np.logspace(np.log10(x_min), np.log10(x_max), 320)
    dash_span = max((x_max - x_min) * float(dash_span_ratio), 1e-8)

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = color_cycle.by_key().get("color", []) if color_cycle is not None else []
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    legend_handles: List[Line2D] = []
    for i, col in enumerate(model_cols):
        color = colors[i % len(colors)]
        y_vals = [float(r[col]) for r in selected]
        all_y_vals.extend(y_vals)

        fit_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x <= 0.5]
        fit_pairs = fit_pairs[:4]
        if len(fit_pairs) < 2:
            fit_pairs = list(zip(x_vals[:4], y_vals[:4]))
        fit_x = [p[0] for p in fit_pairs]
        fit_y = [p[1] for p in fit_pairs]

        poly = fit_curve_logx_for_lambda_b(fit_x, fit_y, degree=int(fit_degree))
        y_fit = poly(np.log10(x_fit))
        y_fit = np.clip(y_fit, 0.0, 1.0)

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

        below_mask = y_fit < y_line
        dash_l = max(x_min, x_join - dash_span)
        dash_r = min(x_max, x_join + dash_span)
        dashed_mask = below_mask & (x_fit >= dash_l) & (x_fit <= dash_r)
        y_fit_solid = np.where(~below_mask, y_fit, np.nan)
        y_fit_dashed = np.where(dashed_mask, y_fit, np.nan)
        ax.plot(x_fit, y_fit_solid, linestyle="-", linewidth=2.2, color=color)
        ax.plot(x_fit, y_fit_dashed, linestyle="--", linewidth=2.2, color=color)

        ax.scatter(x_vals, y_vals, s=28, alpha=0.9, color=color)
        legend_handles.append(Line2D([0], [0], color=color, linestyle="-", linewidth=2.2, label=col))

    if all_y_vals:
        y_min = max(0.0, min(all_y_vals) - 0.02)
        y_max = min(1.0, max(all_y_vals) + 0.02)
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)

    ax.set_xlim(x_min, x_max)
    ax.set_xticks([0.1, 0.2, 0.3, 0.5, 1.0, 2.0])
    ax.set_xticklabels(["0.1", "0.2", "0.3", "0.5", "1.0", "2.0"])

    ax.set_xlabel(r"$\mathit{\lambda}_{\mathit{b}}$", fontsize=14)
    ax.set_ylabel("Macro-F1 Score", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    return legend_handles


def draw_lambda_cons_panel(
    ax,
    selected: List[Dict[str, float]],
    lambda_cons_col: str,
    model_cols: List[str],
    fit_degree: int,
    dash_span_ratio: float,
) -> List[Line2D]:
    x_vals = [float(r[lambda_cons_col]) for r in selected]
    all_y_vals: List[float] = []

    ax.axvspan(0.1, 0.15, color="#f8c471", alpha=0.22)
    ax.text(
        0.102,
        0.985,
        "Recommended region",
        transform=ax.get_xaxis_transform(),
        fontsize=14,  # original 10 -> +4
        va="top",
        color="#7d6608",
    )

    x_min, x_max = 0.05, 0.3
    x_fit = np.linspace(x_min, x_max, 320)
    dash_span = max((x_max - x_min) * float(dash_span_ratio), 1e-8)

    color_cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = color_cycle.by_key().get("color", []) if color_cycle is not None else []
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5"]

    legend_handles: List[Line2D] = []
    for i, col in enumerate(model_cols):
        color = colors[i % len(colors)]
        y_vals = [float(r[col]) for r in selected]
        all_y_vals.extend(y_vals)

        fit_pairs = [(x, y) for x, y in zip(x_vals, y_vals) if x <= 0.2]
        fit_pairs = fit_pairs[:4]
        if len(fit_pairs) < 2:
            fit_pairs = list(zip(x_vals[:4], y_vals[:4]))
        fit_x = [p[0] for p in fit_pairs]
        fit_y = [p[1] for p in fit_pairs]

        poly = fit_curve_linear_for_lambda_cons(fit_x, fit_y, degree=int(fit_degree))
        y_fit = poly(x_fit)
        y_fit = np.clip(y_fit, 0.0, 1.0)

        y1 = find_y_for_x(x_vals, y_vals, 0.3)
        y2 = find_y_for_x(x_vals, y_vals, 0.5)
        slope = (y2 - y1) / (0.5 - 0.3)
        y_line = y1 + slope * (x_fit - 0.3)
        x_join = find_intersection_x(x_fit, y_fit, y_line, x_max=0.3)
        y_join = y1 + slope * (x_join - 0.3)

        if x_min < x_join:
            x_ext_left = np.linspace(x_min, x_join, 120)
            y_ext_left = y1 + slope * (x_ext_left - 0.3)
            ax.plot(x_ext_left, y_ext_left, linestyle="--", linewidth=1.6, alpha=0.9, color=color)
        ax.plot([x_join, 0.5], [y_join, y2], linestyle="-", linewidth=2.2, color=color)

        below_mask = y_fit < y_line
        dash_l = max(x_min, x_join - dash_span)
        dash_r = min(x_max, x_join + dash_span)
        dashed_mask = below_mask & (x_fit >= dash_l) & (x_fit <= dash_r)
        y_fit_solid = np.where(~below_mask, y_fit, np.nan)
        y_fit_dashed = np.where(dashed_mask, y_fit, np.nan)
        ax.plot(x_fit, y_fit_solid, linestyle="-", linewidth=2.2, color=color)
        ax.plot(x_fit, y_fit_dashed, linestyle="--", linewidth=2.2, color=color)
        ax.scatter(x_vals, y_vals, s=28, alpha=0.9, color=color)
        legend_handles.append(Line2D([0], [0], color=color, linestyle="-", linewidth=2.2, label=col))

    if all_y_vals:
        y_min = max(0.0, min(all_y_vals) - 0.02)
        y_max = min(1.0, max(all_y_vals) + 0.02)
        if y_max > y_min:
            ax.set_ylim(y_min, y_max)

    ax.set_xlim(x_min, x_max)
    ax.set_xticks([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
    ax.set_xticklabels(["0.05", "0.1", "0.15", "0.2", "0.25", "0.3"])

    ax.set_xlabel(r"$\mathit{\lambda}_{\mathrm{cons}}$", fontsize=14)
    ax.set_ylabel("Parent-F1 Score", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.35)
    return legend_handles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table_md",
        type=str,
        default="",
        help="Path to markdown table file. If omitted, script auto-searches f1_curve_table.md.",
    )
    ap.add_argument("--lambda_c", type=float, default=None, help="Filter a specific λc value.")
    ap.add_argument("--lambda_b", type=float, default=None, help="Filter a specific λb value for λcons panel.")
    ap.add_argument("--fit_degree_b", type=int, default=3)
    ap.add_argument("--fit_degree_cons", type=int, default=2)
    ap.add_argument("--dash_span_ratio", type=float, default=0.08)
    ap.add_argument("--font_path", type=str, default="")
    ap.add_argument("--out_png", type=str, default="")
    ap.add_argument("--out_svg", type=str, default="")
    args, unknown = ap.parse_known_args()
    if unknown:
        print(f"[WARN] ignored unknown args: {unknown}")

    if int(args.fit_degree_b) < 1 or int(args.fit_degree_cons) < 1:
        raise ValueError("fit degree must be >= 1")
    if not (0.0 < float(args.dash_span_ratio) <= 1.0):
        raise ValueError("--dash_span_ratio must be in (0, 1].")

    table_path = resolve_table_md(args.table_md)
    print(f"[INFO] table_md: {table_path}")
    tables = parse_markdown_tables(table_path)
    if not tables:
        raise ValueError(f"no markdown table found in: {table_path}")

    rows_b = pick_table_for_lambda_b(tables)
    rows_cons = pick_table_for_lambda_cons(tables)

    # Resolve λc for both panels.
    unique_lc_b = sorted({try_float(r[list(rows_b[0].keys())[0]]) for r in rows_b})
    unique_lc_cons = sorted({try_float(r[list(rows_cons[0].keys())[0]]) for r in rows_cons})
    unique_lc_all = sorted(set(unique_lc_b).intersection(set(unique_lc_cons)))
    if args.lambda_c is not None:
        target_lc = float(args.lambda_c)
    else:
        if len(unique_lc_all) > 1:
            raise ValueError(f"multiple λc values found {unique_lc_all}. Please set --lambda_c.")
        target_lc = unique_lc_all[0]

    lambda_b_col, model_cols_b, selected_b = preprocess_lambda_b_rows(rows_b, target_lc)

    # Resolve λb for λcons panel.
    headers_cons = list(rows_cons[0].keys())
    lambda_c_col_cons = headers_cons[0]
    lambda_b_col_cons = headers_cons[1]
    for r in rows_cons:
        r[lambda_c_col_cons] = try_float(r[lambda_c_col_cons])
        r[lambda_b_col_cons] = try_float(r[lambda_b_col_cons])
    unique_lb = sorted({r[lambda_b_col_cons] for r in rows_cons if abs(r[lambda_c_col_cons] - target_lc) < 1e-9})
    if args.lambda_b is not None:
        target_lb = float(args.lambda_b)
    else:
        if len(unique_lb) > 1:
            raise ValueError(f"multiple λb values found {unique_lb}. Please set --lambda_b.")
        target_lb = unique_lb[0]

    lambda_cons_col, model_cols_cons, selected_cons = preprocess_lambda_cons_rows(rows_cons, target_lc, target_lb)
    if model_cols_b != model_cols_cons:
        print("[WARN] model columns differ between two panels; use left-panel legend order.")

    # Global font size +4 relative to original defaults.
    plt.rcParams["font.size"] = 14
    font_used = configure_times_new_roman(args.font_path)

    # Narrower than two independent 6.6-wide panels.
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11.4, 4.8))

    handles_left = draw_lambda_b_panel(
        ax_left,
        selected_b,
        lambda_b_col=lambda_b_col,
        model_cols=model_cols_b,
        fit_degree=int(args.fit_degree_b),
        dash_span_ratio=float(args.dash_span_ratio),
    )
    handles_right = draw_lambda_cons_panel(
        ax_right,
        selected_cons,
        lambda_cons_col=lambda_cons_col,
        model_cols=model_cols_cons,
        fit_degree=int(args.fit_degree_cons),
        dash_span_ratio=float(args.dash_span_ratio),
    )

    # Shared legend, one row at the very bottom.
    legend_handles = handles_left if handles_left else handles_right
    fig.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=13,  # original 9 -> +4
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
        ncol=max(1, len(legend_handles)),
    )

    fig.tight_layout(rect=(0, 0.10, 1, 1))

    if args.out_png.strip():
        out_png = Path(args.out_png).resolve()
    else:
        out_png = table_path.with_name("lambda_b_cons_sensitivity_dual_panel.png")
    if args.out_svg.strip():
        out_svg = Path(args.out_svg).resolve()
    else:
        out_svg = table_path.with_name("lambda_b_cons_sensitivity_dual_panel.svg")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"[INFO] Font used: {font_used}")
    if font_used.lower() not in {"times new roman", "timesnewroman", "times new roman mt", "times"}:
        print("[WARN] Times New Roman not found. Use --font_path to provide a TTF/OTF file if needed.")
    print(f"[OK] PNG saved: {out_png}")
    print(f"[OK] SVG saved: {out_svg}")


if __name__ == "__main__":
    main()
