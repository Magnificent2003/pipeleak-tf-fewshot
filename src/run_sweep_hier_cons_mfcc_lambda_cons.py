import argparse
import csv
import datetime as dt
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import config as cfg


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_float_list(text: str) -> List[float]:
    vals: List[float] = []
    for p in text.split(","):
        s = p.strip()
        if not s:
            continue
        vals.append(float(s))
    if not vals:
        raise ValueError("empty float list")
    return vals


def parse_int_list(text: str) -> List[int]:
    vals: List[int] = []
    for p in text.split(","):
        s = p.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("empty int list")
    return vals


def default_seed_list() -> List[int]:
    seeds: List[int] = []
    for i in range(10):
        name = f"SEED_EXP{i}"
        if hasattr(cfg, name):
            seeds.append(int(getattr(cfg, name)))
    if len(seeds) == 10:
        return seeds
    base = int(getattr(cfg, "SEED", 42))
    return [base + i for i in range(10)]


def choose_python_exe(cli_python: str, project_root: Path) -> str:
    if cli_python.strip():
        return cli_python

    # Windows venv
    py_win = project_root / ".venv" / "Scripts" / "python.exe"
    if py_win.exists():
        return str(py_win)
    # Linux/macOS venv
    py_unix = project_root / ".venv" / "bin" / "python"
    if py_unix.exists():
        return str(py_unix)
    return sys.executable


def safe_float(v, default=float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return default


def parse_best_row(metrics_csv: Path):
    with open(metrics_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = [r for r in rows if str(r.get("stage", "")) == "best"]
    return best[-1] if best else None


def summarize(vals: Sequence[float]) -> Dict[str, float]:
    clean_vals = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not clean_vals:
        return {"mean": float("nan"), "sd": float("nan")}
    m = float(statistics.mean(clean_vals))
    s = float(statistics.stdev(clean_vals)) if len(clean_vals) >= 2 else 0.0
    return {"mean": m, "sd": s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable for child runs.")
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--lambdas", type=str, default="0.05,0.1,0.15,0.2,0.3,0.5")
    ap.add_argument("--seeds", type=str, default="", help="Comma-separated seeds. Empty -> SEED_EXP0~9.")
    ap.add_argument("--cons_loss", type=str, default="sym_kl", choices=["sym_kl", "js", "l2"])
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--keep_stdout", type=int, default=1)
    ap.add_argument("--remove_ckpt", type=int, default=1, help="Remove temporary checkpoint directory at end.")
    ap.add_argument("--continue_on_error", type=int, default=1)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    script_path = src_dir / "run_hier_cons_mfcc_mlp_4cls.py"
    if not script_path.exists():
        raise FileNotFoundError(f"missing script: {script_path}")

    lambdas = parse_float_list(args.lambdas)
    seeds = parse_int_list(args.seeds) if args.seeds.strip() else default_seed_list()

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"sweep_hier_cons_mfcc_lambda_cons_{now_str()}"
    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    ckpt_tmp = out_root / "_tmp_ckpt"
    csv_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.keep_stdout):
        stdout_dir.mkdir(parents=True, exist_ok=True)
    ckpt_tmp.mkdir(parents=True, exist_ok=True)

    python_exe = choose_python_exe(args.python, project_root)
    cont = bool(args.continue_on_error)
    dry = bool(args.dry_run)

    print("========== Sweep (hier_cons MFCC-MLP 4cls) ==========")
    print(f"python      : {python_exe}")
    print(f"script      : {script_path}")
    print(f"out_root    : {out_root}")
    print(f"csv_dir     : {csv_dir}")
    print(f"stdout_dir  : {stdout_dir if bool(args.keep_stdout) else '(disabled)'}")
    print(f"tmp_ckpt    : {ckpt_tmp}")
    print(f"lambdas     : {lambdas}")
    print(f"seeds       : {seeds}")
    print(f"cons_loss   : {args.cons_loss}")
    print("=====================================================")

    total = len(lambdas) * len(seeds)
    run_rows: List[Dict[str, str]] = []
    idx = 0

    for lam in lambdas:
        for run_i, seed in enumerate(seeds, start=1):
            idx += 1
            run_tag = f"lam{lam:g}_run{run_i:02d}_seed{seed}"
            print(f"[{idx:02d}/{total}] START {run_tag}")

            before = {p.resolve() for p in csv_dir.glob("metrics_mlp_mfcc_hier_cons_4cls_*.csv")}
            cmd = [
                python_exe,
                str(script_path),
                "--lambda_cons",
                str(lam),
                "--cons_loss",
                args.cons_loss,
                "--seed",
                str(seed),
                "--save_dir",
                str(ckpt_tmp),
                "--log_dir",
                str(csv_dir),
                "--num_workers",
                str(args.num_workers),
            ]

            status = "ok"
            err = ""
            out_csv = ""
            m_f1 = float("nan")
            p_f1 = float("nan")

            if dry:
                print("[DRY] " + " ".join(cmd))
            else:
                if bool(args.keep_stdout):
                    run_log = stdout_dir / f"{run_tag}.log"
                    with open(run_log, "w", encoding="utf-8") as f:
                        f.write("[CMD] " + " ".join(cmd) + "\n\n")
                        ret = subprocess.run(cmd, cwd=str(project_root), stdout=f, stderr=subprocess.STDOUT, check=False)
                else:
                    ret = subprocess.run(cmd, cwd=str(project_root), check=False)

                if ret.returncode != 0:
                    status = "fail"
                    err = f"return_code={ret.returncode}"
                else:
                    after = list(csv_dir.glob("metrics_mlp_mfcc_hier_cons_4cls_*.csv"))
                    new_csv = [p for p in after if p.resolve() not in before]
                    if not new_csv:
                        status = "fail"
                        err = "no_new_metrics_csv"
                    else:
                        p = sorted(new_csv, key=lambda x: x.stat().st_mtime)[-1]
                        renamed = csv_dir / f"{run_tag}__{p.name}"
                        k = 1
                        while renamed.exists():
                            renamed = csv_dir / f"{run_tag}__{k:02d}__{p.name}"
                            k += 1
                        p.rename(renamed)
                        out_csv = str(renamed)

                        best = parse_best_row(renamed)
                        if best is None:
                            status = "fail"
                            err = "missing_best_row"
                        else:
                            m_f1 = safe_float(best.get("test_macro_f1", ""))
                            p_f1 = safe_float(best.get("test_parent_f1", ""))

            run_rows.append(
                {
                    "lambda_cons": str(lam),
                    "run_index": str(run_i),
                    "seed": str(seed),
                    "status": status,
                    "error": err,
                    "metrics_csv": out_csv,
                    "test_macro_f1": f"{m_f1}",
                    "test_parent_f1": f"{p_f1}",
                }
            )

            if status == "ok":
                print(f"[{idx:02d}/{total}] DONE {run_tag} | macro_f1={m_f1:.4f} | parent_f1={p_f1:.4f}")
            else:
                print(f"[{idx:02d}/{total}] FAIL {run_tag} | {err}")
                if not cont:
                    raise RuntimeError(err)

    raw_csv = csv_dir / "sweep_runs_raw.csv"
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        fields = ["lambda_cons", "run_index", "seed", "status", "error", "metrics_csv", "test_macro_f1", "test_parent_f1"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in run_rows:
            w.writerow(r)

    summary_rows = []
    for lam in lambdas:
        ok_rows = [
            r
            for r in run_rows
            if r["status"] == "ok" and math.isfinite(safe_float(r["lambda_cons"])) and abs(safe_float(r["lambda_cons"]) - lam) < 1e-12
        ]
        macro_vals = [safe_float(r["test_macro_f1"]) for r in ok_rows]
        parent_vals = [safe_float(r["test_parent_f1"]) for r in ok_rows]
        macro_s = summarize(macro_vals)
        parent_s = summarize(parent_vals)
        summary_rows.append(
            {
                "lambda_cons": lam,
                "n_ok": len(ok_rows),
                "n_total": len(seeds),
                "macro_f1_mean": macro_s["mean"],
                "macro_f1_sd": macro_s["sd"],
                "macro_f1_mean_pm_sd": f"{macro_s['mean']:.4f} ± {macro_s['sd']:.4f}",
                "parent_f1_mean": parent_s["mean"],
                "parent_f1_sd": parent_s["sd"],
                "parent_f1_mean_pm_sd": f"{parent_s['mean']:.4f} ± {parent_s['sd']:.4f}",
            }
        )

    summary_csv = csv_dir / "summary_macro_parent_f1_mean_sd.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fields = [
            "lambda_cons",
            "n_ok",
            "n_total",
            "macro_f1_mean",
            "macro_f1_sd",
            "macro_f1_mean_pm_sd",
            "parent_f1_mean",
            "parent_f1_sd",
            "parent_f1_mean_pm_sd",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    if bool(args.remove_ckpt) and ckpt_tmp.exists():
        shutil.rmtree(ckpt_tmp, ignore_errors=True)

    print("============== Done ==============")
    print(f"raw runs    : {raw_csv}")
    print(f"summary     : {summary_csv}")
    print(f"stdout logs : {stdout_dir if bool(args.keep_stdout) else '(disabled)'}")
    print(f"ckpt dir    : {'removed' if bool(args.remove_ckpt) else ckpt_tmp}")
    print("==================================")


if __name__ == "__main__":
    main()
