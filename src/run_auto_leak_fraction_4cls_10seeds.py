import argparse
import csv
import datetime as dt
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import config as cfg


FRACTIONS = [1.0, 0.8, 0.6, 0.4]


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


def parse_fractions(text: str) -> List[float]:
    out: List[float] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        v = float(p)
        if v <= 0 or v > 1:
            raise ValueError(f"fraction must be in (0,1], got {v}")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("empty --fractions")
    return out


def newest_path(paths: Sequence[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate file found.")
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def pick_new_file(dir_path: Path, pattern: str, before: Sequence[Path]) -> Path:
    before_set = {p.resolve() for p in before}
    after = list(dir_path.glob(pattern))
    new_files = [p for p in after if p.resolve() not in before_set]
    if not new_files:
        raise FileNotFoundError(f"No new file matched {pattern} in {dir_path}")
    return newest_path(new_files)


def tag_csv_name(csv_path: Path, seed_name: str, seed_value: int, frac: float, model_tag: str) -> Path:
    frac_tag = f"fr_{frac:.1f}".replace(".", "p")
    prefix = f"{seed_name}_{seed_value}__{frac_tag}__{model_tag}__"
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
        "valve_recall": _get("test_valve_recall"),
        "valve_f1": _get("test_valve_f1"),
        "cross_parent_error_rate": _get("test_cross_parent_err"),
        "parent_leak_recall": _get("test_parent_leak_recall"),
    }


def dump_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def summarize_mean_sd(detail_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    groups: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for r in detail_rows:
        key = (r["fraction"], r["model"])
        groups.setdefault(key, []).append(r)

    out: List[Dict[str, str]] = []
    for (fraction, model), rows in sorted(groups.items(), key=lambda x: (float(x[0][0]), x[0][1]), reverse=True):
        def _arr(name: str) -> np.ndarray:
            return np.array([float(r[name]) for r in rows], dtype=np.float64)

        a_vr = _arr("valve_recall")
        a_vf1 = _arr("valve_f1")
        a_cpe = _arr("cross_parent_error_rate")
        a_plr = _arr("parent_leak_recall")

        def _mean_sd(a: np.ndarray) -> Tuple[float, float, str]:
            mean = float(np.mean(a)) if a.size else float("nan")
            sd = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
            return mean, sd, f"{mean:.6f} ± {sd:.6f}"

        vr_m, vr_sd, vr_fmt = _mean_sd(a_vr)
        vf_m, vf_sd, vf_fmt = _mean_sd(a_vf1)
        cpe_m, cpe_sd, cpe_fmt = _mean_sd(a_cpe)
        plr_m, plr_sd, plr_fmt = _mean_sd(a_plr)

        out.append(
            {
                "fraction": fraction,
                "model": model,
                "n_seeds": str(len(rows)),
                "valve_recall_mean": f"{vr_m:.6f}",
                "valve_recall_sd": f"{vr_sd:.6f}",
                "valve_recall_mean_pm_sd": vr_fmt,
                "valve_f1_mean": f"{vf_m:.6f}",
                "valve_f1_sd": f"{vf_sd:.6f}",
                "valve_f1_mean_pm_sd": vf_fmt,
                "cross_parent_error_rate_mean": f"{cpe_m:.6f}",
                "cross_parent_error_rate_sd": f"{cpe_sd:.6f}",
                "cross_parent_error_rate_mean_pm_sd": cpe_fmt,
                "parent_leak_recall_mean": f"{plr_m:.6f}",
                "parent_leak_recall_sd": f"{plr_sd:.6f}",
                "parent_leak_recall_mean_pm_sd": plr_fmt,
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable used to run child scripts.")
    ap.add_argument("--out_root", type=str, default="", help="Output root directory.")
    ap.add_argument("--seed_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--fractions", type=str, default="1.0,0.8,0.6,0.4")
    ap.add_argument("--save_ckpt", type=int, default=0, help="1=keep checkpoints; 0=delete temp checkpoints after run")
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = args.python.strip() if args.python.strip() else choose_python_exe(project_root)
    dry_run = bool(args.dry_run)

    all_seeds = get_seed_list()
    seed_indices = parse_seed_indices(args.seed_indices)
    seed_list = [all_seeds[i] for i in seed_indices]
    fractions = parse_fractions(args.fractions)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"auto_leak_fraction_4cls_{now_str()}"
    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    ckpt_dir = out_root / "checkpoints"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    detail_csv = csv_dir / "detail_seed_fraction_model_metrics.csv"
    summary_csv = csv_dir / "summary_mean_sd_by_fraction_model.csv"
    step_log_csv = csv_dir / "step_log.csv"

    detail_fields = [
        "seed_name",
        "seed_value",
        "fraction",
        "model",
        "valve_recall",
        "valve_f1",
        "cross_parent_error_rate",
        "parent_leak_recall",
        "csv_file",
    ]
    step_fields = [
        "seed_name",
        "seed_value",
        "fraction",
        "step",
        "status",
        "duration_sec",
        "error",
        "start_time",
        "end_time",
    ]

    detail_rows: List[Dict[str, str]] = []
    step_rows: List[Dict[str, str]] = []

    print("========== Auto Leak-Fraction 4cls (10 seeds) ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print(f"seed_list  : {seed_list}")
    print(f"fractions  : {fractions}")
    print(f"save_ckpt  : {args.save_ckpt}")
    print(f"dry_run    : {dry_run}")
    print("========================================================")

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
                    "fraction": "",
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

        for frac in fractions:
            print(f"\n-- {seed_tag} | leak_fraction={frac:.1f} --")
            frac_tag = f"fr_{frac:.1f}".replace(".", "p")
            frac_ckpt_dir = ckpt_dir / seed_tag / frac_tag
            frac_ckpt_dir.mkdir(parents=True, exist_ok=True)

            ckpt_dark_ce = ""
            ckpt_mfcc_ce = ""

            jobs = [
                {
                    "step": "darknet4_ce",
                    "script": "run_darknet_4cls_imbalance.py",
                    "args": [
                        "--data_root", str(cfg.DATASET_STFT),
                        "--save_dir", str(frac_ckpt_dir),
                        "--log_dir", str(csv_dir),
                        "--save_ckpt", "1",
                        "--seed", str(seed_value),
                        "--loss_type", "ce",
                        "--subset_mode", "leak_fraction",
                        "--leak_fraction", f"{frac}",
                    ],
                    "csv_pat": "metrics_darknet19_4cls_imbalance_ce_*.csv",
                    "ckpt_pat": "darknet19_4cls_imbalance_ce_best_*.pth",
                    "model_name": "DarkNet-19-4cls",
                    "model_tag": "darknet4_ce",
                    "need_ckpt": True,
                },
                {
                    "step": "mfcc4_ce",
                    "script": "run_mfcc_mlp_4cls_imbalance.py",
                    "args": [
                        "--data_root", str(cfg.DATASET),
                        "--save_dir", str(frac_ckpt_dir),
                        "--log_dir", str(csv_dir),
                        "--save_ckpt", "1",
                        "--seed", str(seed_value),
                        "--loss_type", "ce",
                        "--subset_mode", "leak_fraction",
                        "--leak_fraction", f"{frac}",
                    ],
                    "csv_pat": "metrics_mlp_mfcc_4cls_imbalance_ce_*.csv",
                    "ckpt_pat": "mlp_mfcc_4cls_imbalance_ce_best_*.pt",
                    "model_name": "MFCC-MLP-4cls",
                    "model_tag": "mfcc4_ce",
                    "need_ckpt": True,
                },
                {
                    "step": "fuse_cwgf",
                    "script": "run_fuse_mfcc_darknet_cwgf_4cls.py",
                    "args": [
                        "--data_root_raw", str(cfg.DATASET),
                        "--data_root_stft", str(cfg.DATASET_STFT),
                        "--save_dir", str(frac_ckpt_dir),
                        "--log_dir", str(csv_dir),
                        "--save_ckpt", "0",
                        "--seed", str(seed_value),
                        "--subset_mode", "leak_fraction",
                        "--leak_fraction", f"{frac}",
                    ],
                    "csv_pat": "metrics_fuse_cwgf_mfcc_darknet_4cls_*.csv",
                    "ckpt_pat": "",
                    "model_name": "CWGF (DarkNet-19 & MFCC-MLP)",
                    "model_tag": "cwgf",
                    "need_ckpt": False,
                },
                {
                    "step": "darknet4_weighted",
                    "script": "run_darknet_4cls_imbalance.py",
                    "args": [
                        "--data_root", str(cfg.DATASET_STFT),
                        "--save_dir", str(frac_ckpt_dir),
                        "--log_dir", str(csv_dir),
                        "--save_ckpt", "0",
                        "--seed", str(seed_value),
                        "--loss_type", "weighted_ce",
                        "--subset_mode", "leak_fraction",
                        "--leak_fraction", f"{frac}",
                    ],
                    "csv_pat": "metrics_darknet19_4cls_imbalance_weighted_ce_*.csv",
                    "ckpt_pat": "",
                    "model_name": "DarkNet-19-4cls (Weighted-CE)",
                    "model_tag": "darknet4_weighted",
                    "need_ckpt": False,
                },
            ]

            for job in jobs:
                t0 = time.time()
                ts0 = dt.datetime.now().isoformat(timespec="seconds")
                status = "ok"
                err = ""
                try:
                    cmd = [python_exe, job["script"]] + job["args"]
                    if job["step"] == "fuse_cwgf":
                        if (not dry_run) and (not ckpt_dark_ce or not ckpt_mfcc_ce):
                            raise RuntimeError("Missing CE expert checkpoints for CWGF fusion.")
                        cmd += [
                            "--ckpt_darknet4", ckpt_dark_ce if ckpt_dark_ce else "DUMMY_DARK.pth",
                            "--ckpt_mfcc4", ckpt_mfcc_ce if ckpt_mfcc_ce else "DUMMY_MFCC.pt",
                        ]

                    before_csv = list(csv_dir.glob(job["csv_pat"]))
                    before_ckpt = list(frac_ckpt_dir.glob(job["ckpt_pat"])) if job["ckpt_pat"] else []

                    run_cmd(
                        cmd,
                        cwd=src_dir,
                        stdout_file=stdout_dir / f"{seed_tag}__{frac_tag}__{job['step']}.log",
                        dry_run=dry_run,
                    )

                    if not dry_run:
                        new_csv = pick_new_file(csv_dir, job["csv_pat"], before_csv)
                        tagged_csv = tag_csv_name(new_csv, seed_name, seed_value, frac, job["model_tag"])
                        mets = read_best_metrics(tagged_csv)
                        detail_rows.append(
                            {
                                "seed_name": seed_name,
                                "seed_value": str(seed_value),
                                "fraction": f"{frac:.1f}",
                                "model": job["model_name"],
                                "valve_recall": f"{mets['valve_recall']:.6f}",
                                "valve_f1": f"{mets['valve_f1']:.6f}",
                                "cross_parent_error_rate": f"{mets['cross_parent_error_rate']:.6f}",
                                "parent_leak_recall": f"{mets['parent_leak_recall']:.6f}",
                                "csv_file": tagged_csv.name,
                            }
                        )
                        dump_csv(detail_csv, detail_fields, detail_rows)

                        if job["need_ckpt"]:
                            new_ckpt = pick_new_file(frac_ckpt_dir, job["ckpt_pat"], before_ckpt)
                            if job["step"] == "darknet4_ce":
                                ckpt_dark_ce = str(new_ckpt.resolve())
                            elif job["step"] == "mfcc4_ce":
                                ckpt_mfcc_ce = str(new_ckpt.resolve())
                except Exception as e:
                    status = "fail"
                    err = str(e)

                t1 = time.time()
                ts1 = dt.datetime.now().isoformat(timespec="seconds")
                step_rows.append(
                    {
                        "seed_name": seed_name,
                        "seed_value": str(seed_value),
                        "fraction": f"{frac:.1f}",
                        "step": job["step"],
                        "status": status,
                        "duration_sec": f"{(t1 - t0):.2f}",
                        "error": err,
                        "start_time": ts0,
                        "end_time": ts1,
                    }
                )
                dump_csv(step_log_csv, step_fields, step_rows)
                if status == "fail":
                    print(f"[FAIL] {seed_tag} frac={frac:.1f} step={job['step']} | {err}")
                    break

            if (not dry_run) and int(args.save_ckpt) == 0:
                shutil.rmtree(frac_ckpt_dir, ignore_errors=True)

    summary_rows = summarize_mean_sd(detail_rows)
    summary_fields = [
        "fraction",
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
        "parent_leak_recall_mean",
        "parent_leak_recall_sd",
        "parent_leak_recall_mean_pm_sd",
    ]
    dump_csv(summary_csv, summary_fields, summary_rows)

    print("\nAll runs finished.")
    print(f"Detail : {detail_csv}")
    print(f"Summary: {summary_csv}")
    print(f"StepLog: {step_log_csv}")


if __name__ == "__main__":
    main()
