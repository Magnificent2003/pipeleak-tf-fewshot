import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import config as cfg


LOSSES = ["ce", "weighted_ce", "focal"]


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


def run_cmd(cmd: Sequence[str], cwd: Path, stdout_file: Path, dry_run: bool = False) -> None:
    print(f"[CMD] (cwd={cwd}) {' '.join(cmd)}")
    if dry_run:
        return
    stdout_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_file, "w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        ret = subprocess.run(list(cmd), cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"return_code={ret.returncode}")


def get_seed_list() -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for i in range(10):
        name = f"SEED_EXP{i}"
        if not hasattr(cfg, name):
            raise AttributeError(f"Missing {name} in config.py")
        out.append((name, int(getattr(cfg, name))))
    return out


def parse_seed_indices(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        idx = int(p)
        if idx < 0 or idx > 9:
            raise ValueError(f"seed index out of range [0,9]: {idx}")
        if idx not in out:
            out.append(idx)
    if not out:
        raise ValueError("empty --seed_indices")
    return out


def parse_losses(text: str) -> List[str]:
    out: List[str] = []
    for part in text.split(","):
        p = part.strip().lower()
        if not p:
            continue
        if p not in LOSSES:
            raise ValueError(f"Unsupported loss '{p}', choose from {LOSSES}")
        if p not in out:
            out.append(p)
    if not out:
        raise ValueError("empty --losses")
    return out


def newest_path(paths: Sequence[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate file found.")
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def pick_new_csv(csv_dir: Path, pattern: str, before: Sequence[Path]) -> Path:
    before_set = {p.resolve() for p in before}
    after = list(csv_dir.glob(pattern))
    new_files = [p for p in after if p.resolve() not in before_set]
    if not new_files:
        raise FileNotFoundError(f"No new csv matched {pattern} in {csv_dir}")
    return newest_path(new_files)


def tag_csv_name(csv_path: Path, seed_name: str, seed_value: int, loss_name: str) -> Path:
    prefix = f"{seed_name}_{seed_value}__{loss_name}__"
    target = csv_path.with_name(prefix + csv_path.name)
    k = 1
    while target.exists():
        target = csv_path.with_name(prefix + f"{k:02d}__" + csv_path.name)
        k += 1
    csv_path.rename(target)
    return target


def read_best_metrics(csv_path: Path) -> Dict[str, float]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty csv: {csv_path}")
    best_rows = [r for r in rows if str(r.get("epoch", "")).strip().lower() == "best"]
    row = best_rows[-1] if best_rows else rows[-1]

    def _get(name: str) -> float:
        v = str(row.get(name, "")).strip()
        if v == "":
            raise ValueError(f"Missing {name} in {csv_path}")
        return float(v)

    return {
        "test_valve_recall": _get("test_valve_recall"),
        "test_valve_f1": _get("test_valve_f1"),
        "test_cross_parent_err": _get("test_cross_parent_err"),
    }


def dump_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_mean_sd(detail_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    model_names = sorted({r["model"] for r in detail_rows})
    out: List[Dict[str, str]] = []
    for m in model_names:
        vals_vr = []
        vals_vf1 = []
        vals_cpe = []
        for r in detail_rows:
            if r["model"] != m:
                continue
            vals_vr.append(float(r["valve_recall"]))
            vals_vf1.append(float(r["valve_f1"]))
            vals_cpe.append(float(r["cross_parent_error_rate"]))

        arr_vr = np.array(vals_vr, dtype=np.float64)
        arr_vf1 = np.array(vals_vf1, dtype=np.float64)
        arr_cpe = np.array(vals_cpe, dtype=np.float64)

        def _mean_sd(a: np.ndarray) -> Tuple[float, float, str]:
            mean = float(np.mean(a)) if a.size else float("nan")
            sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
            return mean, sd, f"{mean:.6f} ± {sd:.6f}"

        vr_mean, vr_sd, vr_fmt = _mean_sd(arr_vr)
        vf1_mean, vf1_sd, vf1_fmt = _mean_sd(arr_vf1)
        cpe_mean, cpe_sd, cpe_fmt = _mean_sd(arr_cpe)

        out.append(
            {
                "model": m,
                "n_seeds": str(len(arr_vr)),
                "valve_recall_mean": f"{vr_mean:.6f}",
                "valve_recall_sd": f"{vr_sd:.6f}",
                "valve_recall_mean_pm_sd": vr_fmt,
                "valve_f1_mean": f"{vf1_mean:.6f}",
                "valve_f1_sd": f"{vf1_sd:.6f}",
                "valve_f1_mean_pm_sd": vf1_fmt,
                "cross_parent_error_rate_mean": f"{cpe_mean:.6f}",
                "cross_parent_error_rate_sd": f"{cpe_sd:.6f}",
                "cross_parent_error_rate_mean_pm_sd": cpe_fmt,
                "cross_parent_error_percent_mean": f"{(cpe_mean * 100.0):.4f}",
                "cross_parent_error_percent_sd": f"{(cpe_sd * 100.0):.4f}",
                "cross_parent_error_percent_mean_pm_sd": f"{(cpe_mean*100.0):.4f} ± {(cpe_sd*100.0):.4f}",
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable used to run child scripts.")
    ap.add_argument("--out_root", type=str, default="", help="Output root directory.")
    ap.add_argument("--seed_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--losses", type=str, default="ce,weighted_ce,focal")
    ap.add_argument("--save_ckpt", type=int, default=0)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = args.python.strip() if args.python.strip() else choose_python_exe(project_root)
    dry_run = bool(args.dry_run)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"auto_darknet4_imbalance_10seeds_{now_str()}"
    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    ckpt_dir = out_root / "checkpoints"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = csv_dir / "detail_seed_loss_metrics.csv"
    summary_csv = csv_dir / "summary_mean_sd.csv"
    step_log_csv = csv_dir / "step_log.csv"

    detail_rows: List[Dict[str, str]] = []
    step_rows: List[Dict[str, str]] = []

    step_fields = [
        "seed_name",
        "seed_value",
        "loss_type",
        "step",
        "status",
        "duration_sec",
        "error",
        "start_time",
        "end_time",
    ]
    detail_fields = [
        "seed_name",
        "seed_value",
        "loss_type",
        "model",
        "valve_recall",
        "valve_f1",
        "cross_parent_error_rate",
        "cross_parent_error_percent",
        "csv_file",
    ]

    all_seeds = get_seed_list()
    seed_indices = parse_seed_indices(args.seed_indices)
    seed_list = [all_seeds[i] for i in seed_indices]
    loss_list = parse_losses(args.losses)

    print("========== Auto DarkNet-19 4cls Imbalance (10 seeds) ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print(f"seed_list  : {seed_list}")
    print(f"losses     : {loss_list}")
    print(f"save_ckpt  : {args.save_ckpt}")
    print(f"dry_run    : {dry_run}")
    print("===============================================================")

    for seed_name, seed_value in seed_list:
        seed_tag = f"{seed_name}_{seed_value}"
        print(f"\n==== Start seed: {seed_tag} ====")

        build_steps = [
            (
                "build_datasets",
                [
                    python_exe,
                    "-c",
                    f"import config as cfg; cfg.SEED={seed_value}; import build_datasets as m; m.main()",
                ],
            ),
            ("build_stft_datasets", [python_exe, "build_stft_datasets.py"]),
        ]

        seed_failed = False
        for step_name, cmd in build_steps:
            t0 = time.time()
            ts0 = dt.datetime.now().isoformat(timespec="seconds")
            status = "ok"
            err = ""
            try:
                run_cmd(cmd, cwd=src_dir, stdout_file=stdout_dir / f"{seed_tag}__{step_name}.log", dry_run=dry_run)
            except Exception as e:
                status = "fail"
                err = str(e)
                seed_failed = True
            t1 = time.time()
            ts1 = dt.datetime.now().isoformat(timespec="seconds")
            step_rows.append(
                {
                    "seed_name": seed_name,
                    "seed_value": str(seed_value),
                    "loss_type": "",
                    "step": step_name,
                    "status": status,
                    "duration_sec": f"{(t1 - t0):.2f}",
                    "error": err,
                    "start_time": ts0,
                    "end_time": ts1,
                }
            )
            dump_csv(step_log_csv, step_fields, step_rows)
            if seed_failed:
                break

        if seed_failed:
            print(f"[FAIL] {seed_tag} failed in build steps.")
            continue

        for loss_type in loss_list:
            t0 = time.time()
            ts0 = dt.datetime.now().isoformat(timespec="seconds")
            status = "ok"
            err = ""
            try:
                before = list(csv_dir.glob(f"metrics_darknet19_4cls_imbalance_{loss_type}_*.csv"))
                cmd = [
                    python_exe,
                    "run_darknet_4cls_imbalance.py",
                    "--data_root",
                    str(cfg.DATASET_STFT),
                    "--save_dir",
                    str(ckpt_dir),
                    "--log_dir",
                    str(csv_dir),
                    "--save_ckpt",
                    str(int(args.save_ckpt)),
                    "--seed",
                    str(seed_value),
                    "--loss_type",
                    loss_type,
                    "--focal_gamma",
                    "2.0",
                    "--train_counts",
                    "157,32,79,110",
                ]
                run_cmd(
                    cmd,
                    cwd=src_dir,
                    stdout_file=stdout_dir / f"{seed_tag}__train_{loss_type}.log",
                    dry_run=dry_run,
                )
                if not dry_run:
                    new_csv = pick_new_csv(csv_dir, f"metrics_darknet19_4cls_imbalance_{loss_type}_*.csv", before)
                    tagged_csv = tag_csv_name(new_csv, seed_name, seed_value, loss_type)
                    mets = read_best_metrics(tagged_csv)

                    model_name = {
                        "ce": "DarkNet-19 + CE",
                        "weighted_ce": "DarkNet-19 + Weighted-CE",
                        "focal": "DarkNet-19 + Focal",
                    }[loss_type]
                    detail_rows.append(
                        {
                            "seed_name": seed_name,
                            "seed_value": str(seed_value),
                            "loss_type": loss_type,
                            "model": model_name,
                            "valve_recall": f"{mets['test_valve_recall']:.6f}",
                            "valve_f1": f"{mets['test_valve_f1']:.6f}",
                            "cross_parent_error_rate": f"{mets['test_cross_parent_err']:.6f}",
                            "cross_parent_error_percent": f"{(mets['test_cross_parent_err']*100.0):.4f}",
                            "csv_file": tagged_csv.name,
                        }
                    )
                    dump_csv(detail_csv, detail_fields, detail_rows)
            except Exception as e:
                status = "fail"
                err = str(e)

            t1 = time.time()
            ts1 = dt.datetime.now().isoformat(timespec="seconds")
            step_rows.append(
                {
                    "seed_name": seed_name,
                    "seed_value": str(seed_value),
                    "loss_type": loss_type,
                    "step": "train_eval",
                    "status": status,
                    "duration_sec": f"{(t1 - t0):.2f}",
                    "error": err,
                    "start_time": ts0,
                    "end_time": ts1,
                }
            )
            dump_csv(step_log_csv, step_fields, step_rows)

            if status == "fail":
                print(f"[FAIL] {seed_tag} loss={loss_type} | {err}")
            else:
                print(f"[DONE] {seed_tag} loss={loss_type}")

    summary_rows = summarize_mean_sd(detail_rows)
    summary_fields = [
        "model",
        "n_seeds",
        "valve_recall_mean",
        "valve_recall_sd",
        "valve_recall_mean_pm_sd",
        "valve_f1_mean",
        "valve_f1_sd",
        "valve_f1_mean_pm_sd",
        "cross_parent_error_rate_mean",
        "cross_parent_error_rate_sd",
        "cross_parent_error_rate_mean_pm_sd",
        "cross_parent_error_percent_mean",
        "cross_parent_error_percent_sd",
        "cross_parent_error_percent_mean_pm_sd",
    ]
    dump_csv(summary_csv, summary_fields, summary_rows)

    print("\nAll seeds finished.")
    print(f"Detail : {detail_csv}")
    print(f"Summary: {summary_csv}")
    print(f"StepLog: {step_log_csv}")


if __name__ == "__main__":
    main()
