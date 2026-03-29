import argparse
import csv
import datetime as dt
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def choose_python_exe(project_root: Path) -> str:
    py_venv_unix = project_root / ".venv" / "bin" / "python"
    if py_venv_unix.exists():
        return str(py_venv_unix)
    py_venv_win = project_root / ".venv" / "Scripts" / "python.exe"
    if py_venv_win.exists():
        return str(py_venv_win)
    return sys.executable


def run_cmd(cmd: Sequence[str], cwd: Path, stdout_file: Path) -> None:
    print(f"[CMD] (cwd={cwd}) {' '.join(cmd)}")
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_file, "w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        ret = subprocess.run(list(cmd), cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"return_code={ret.returncode}")


def newest_path(paths: Sequence[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate file found.")
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def pick_new_file(dir_path: Path, pattern: str, before: Sequence[Path]) -> Path:
    before_set = {p.resolve() for p in before}
    after = list(dir_path.glob(pattern))
    new_files = [p for p in after if p.resolve() not in before_set]
    if not new_files:
        raise FileNotFoundError(f"No new file matched pattern '{pattern}' in {dir_path}")
    return newest_path(new_files)


def load_val_curve(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or "val_macro_f1" not in df.columns:
        raise ValueError(f"{csv_path} missing epoch/val_macro_f1 columns.")
    df = df.copy()
    df["epoch_num"] = pd.to_numeric(df["epoch"], errors="coerce")
    df["val_macro_f1"] = pd.to_numeric(df["val_macro_f1"], errors="coerce")
    df = df.dropna(subset=["epoch_num", "val_macro_f1"])
    df["epoch_num"] = df["epoch_num"].astype(int)
    df = df.sort_values("epoch_num").drop_duplicates("epoch_num", keep="last")
    return df[["epoch_num", "val_macro_f1"]]


def plot_curves(df_curve: pd.DataFrame, out_png: Path, out_svg: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["axes.linewidth"] = 1.2
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.width"] = 1.2

    styles = {
        "baseline_mfcc_mlp_4cls": {"color": "#111111", "marker": "s", "label": "MFCC-MLP-4cls"},
        "hcl_lambda0.3_start100": {"color": "#e31a1c", "marker": "o", "label": "MFCC-MLP-4cls (HCL, λ=0.3@100)"},
        "hcl_lambda0.5_start0": {"color": "#1f78b4", "marker": "^", "label": "MFCC-MLP-4cls (HCL, λ=0.5@0)"},
        "hcl_lambda0.1_start50": {"color": "#33a02c", "marker": "D", "label": "MFCC-MLP-4cls (HCL, λ=0.1@50)"},
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=300)
    x = df_curve["epoch"].to_numpy(dtype=np.int64)
    for key, st in styles.items():
        y = df_curve[key].to_numpy(dtype=np.float64)
        ax.plot(x, y, color=st["color"], linewidth=2.0, label=st["label"])

    ax.set_xlabel("Epoch", fontsize=16)
    ax.set_ylabel("Val Macro-F1", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_xlim(1, int(np.max(x)))
    ax.minorticks_off()
    ax.legend(frameon=False, fontsize=10, ncol=1, loc="best")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    fig.savefig(out_svg, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2023)
    ap.add_argument("--num_workers", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    py = choose_python_exe(project_root)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"tmp_mfcc_mlp_hcl_valcurve_{now_str()}"
    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    ckpt_tmp = out_root / "tmp_ckpt"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tmp.mkdir(parents=True, exist_ok=True)

    exp_list: List[Dict] = [
        {
            "id": "baseline_mfcc_mlp_4cls",
            "script": "run_base_mfcc_mlp_4cls.py",
            "csv_pattern": "metrics_mlp_mfcc_4cls_*.csv",
            "args": [
                "--epochs", str(args.epochs),
                "--patience", "0",
                "--loss_type", "ce",
                "--seed", str(args.seed),
                "--num_workers", str(args.num_workers),
                "--log_dir", str(csv_dir),
                "--save_dir", str(ckpt_tmp),
            ],
        },
        {
            "id": "hcl_lambda0.3_start100",
            "script": "run_mfcc_mlp_hcl_4cls_sched.py",
            "csv_pattern": "metrics_mlp_mfcc_hcl_sched_4cls_*.csv",
            "args": [
                "--epochs", str(args.epochs),
                "--cons_start_epoch", "100",
                "--lambda_cons", "0.3",
                "--seed", str(args.seed),
                "--num_workers", str(args.num_workers),
                "--save_ckpt", "0",
                "--log_dir", str(csv_dir),
                "--save_dir", str(ckpt_tmp),
            ],
        },
        {
            "id": "hcl_lambda0.5_start0",
            "script": "run_mfcc_mlp_hcl_4cls_sched.py",
            "csv_pattern": "metrics_mlp_mfcc_hcl_sched_4cls_*.csv",
            "args": [
                "--epochs", str(args.epochs),
                "--cons_start_epoch", "0",
                "--lambda_cons", "0.5",
                "--seed", str(args.seed),
                "--num_workers", str(args.num_workers),
                "--save_ckpt", "0",
                "--log_dir", str(csv_dir),
                "--save_dir", str(ckpt_tmp),
            ],
        },
        {
            "id": "hcl_lambda0.1_start50",
            "script": "run_mfcc_mlp_hcl_4cls_sched.py",
            "csv_pattern": "metrics_mlp_mfcc_hcl_sched_4cls_*.csv",
            "args": [
                "--epochs", str(args.epochs),
                "--cons_start_epoch", "50",
                "--lambda_cons", "0.1",
                "--seed", str(args.seed),
                "--num_workers", str(args.num_workers),
                "--save_ckpt", "0",
                "--log_dir", str(csv_dir),
                "--save_dir", str(ckpt_tmp),
            ],
        },
    ]

    collected: Dict[str, Path] = {}
    step_rows: List[Dict[str, str]] = []
    step_fields = ["exp_id", "status", "duration_sec", "error", "start_time", "end_time", "csv_file"]

    print("========== MFCC-MLP-4cls vs HCL (val macro-f1 curves) ==========")
    print(f"python_exe : {py}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print("===============================================================")

    for exp in exp_list:
        exp_id = exp["id"]
        t0 = time.time()
        ts0 = dt.datetime.now().isoformat(timespec="seconds")
        status, err, csv_file = "ok", "", ""
        try:
            before = list(csv_dir.glob(exp["csv_pattern"]))
            cmd = [py, exp["script"]] + exp["args"]
            run_cmd(cmd, cwd=src_dir, stdout_file=stdout_dir / f"{exp_id}.log")
            new_csv = pick_new_file(csv_dir, exp["csv_pattern"], before)
            target = csv_dir / f"{exp_id}__{new_csv.name}"
            new_csv.rename(target)
            collected[exp_id] = target
            csv_file = str(target)
        except Exception as e:
            status, err = "fail", str(e)

        t1 = time.time()
        ts1 = dt.datetime.now().isoformat(timespec="seconds")
        step_rows.append(
            {
                "exp_id": exp_id,
                "status": status,
                "duration_sec": f"{(t1 - t0):.2f}",
                "error": err,
                "start_time": ts0,
                "end_time": ts1,
                "csv_file": csv_file,
            }
        )
        with open(csv_dir / "run_steps.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=step_fields)
            w.writeheader()
            w.writerows(step_rows)

        if status == "fail":
            raise RuntimeError(f"{exp_id} failed: {err}")

    # Merge val_macro_f1 curves
    epoch_axis = pd.DataFrame({"epoch": np.arange(1, int(args.epochs) + 1, dtype=np.int64)})
    merged = epoch_axis.copy()
    for exp in exp_list:
        exp_id = exp["id"]
        df = load_val_curve(collected[exp_id]).rename(columns={"epoch_num": "epoch", "val_macro_f1": exp_id})
        merged = merged.merge(df, on="epoch", how="left")

    merged_csv = csv_dir / "val_macro_f1_curves.csv"
    merged.to_csv(merged_csv, index=False, encoding="utf-8")

    out_png = csv_dir / "val_macro_f1_curves_4exp.png"
    out_svg = csv_dir / "val_macro_f1_curves_4exp.svg"
    plot_curves(merged, out_png=out_png, out_svg=out_svg)

    # remove any weights
    shutil.rmtree(ckpt_tmp, ignore_errors=True)

    print(f"[OK] merged csv: {merged_csv}")
    print(f"[OK] figure png: {out_png}")
    print(f"[OK] figure svg: {out_svg}")
    print(f"[OK] run steps : {csv_dir / 'run_steps.csv'}")


if __name__ == "__main__":
    main()
