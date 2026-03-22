import argparse
import csv
import datetime as dt
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import config as cfg


FRACTIONS: List[float] = [1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]


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


def parse_fraction_list(text: str) -> List[float]:
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


def dump_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def newest_path(paths: Sequence[Path]) -> Path:
    if not paths:
        raise FileNotFoundError("No candidate file found.")
    return sorted(paths, key=lambda p: p.stat().st_mtime)[-1]


def parse_metric_from_csv(csv_path: Path, metric_cols: Sequence[str]) -> float:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty csv: {csv_path}")

    best_rows = [r for r in rows if str(r.get("epoch", "")).strip().lower() == "best"]
    row = best_rows[-1] if best_rows else rows[-1]

    for col in metric_cols:
        v = str(row.get(col, "")).strip()
        if v != "":
            return float(v)
    raise ValueError(f"Cannot parse metrics {metric_cols} from {csv_path}")


def frac_tag(frac: float) -> str:
    return f"fr_{frac:.1f}".replace(".", "p")


def tag_csv_name(csv_path: Path, seed_name: str, seed_value: int, frac: float, model_tag: str) -> Path:
    prefix = f"{seed_name}_{seed_value}__{frac_tag(frac)}__{model_tag}__"
    target = csv_path.with_name(prefix + csv_path.name)
    k = 1
    while target.exists():
        target = csv_path.with_name(prefix + f"{k:02d}__" + csv_path.name)
        k += 1
    csv_path.rename(target)
    return target


def snapshot_train_arrays(img_size: int) -> Dict[str, np.ndarray]:
    raw_root = Path(cfg.DATASET)
    stft_root = Path(cfg.DATASET_STFT)

    p_raw_x = raw_root / "X_train.npy"
    p_raw_y = raw_root / "y_train.npy"
    p_raw_y4 = raw_root / "y4_train.npy"
    p_raw_idx = raw_root / "train_idx.npy"

    p_stft_x = stft_root / f"X_train_stft_{img_size}.npy"
    p_stft_y = stft_root / "y_train.npy"
    p_stft_y4 = stft_root / "y4_train.npy"

    required = [p_raw_x, p_raw_y, p_raw_y4, p_stft_x, p_stft_y, p_stft_y4]
    miss = [str(p) for p in required if not p.exists()]
    if miss:
        raise FileNotFoundError("Missing train files for snapshot:\n" + "\n".join(miss))

    snap = {
        "raw_x": np.load(p_raw_x),
        "raw_y": np.load(p_raw_y).astype(np.int64),
        "raw_y4": np.load(p_raw_y4).astype(np.int64),
        "stft_x": np.load(p_stft_x),
        "stft_y": np.load(p_stft_y).astype(np.int64),
        "stft_y4": np.load(p_stft_y4).astype(np.int64),
        "p_raw_x": p_raw_x,
        "p_raw_y": p_raw_y,
        "p_raw_y4": p_raw_y4,
        "p_raw_idx": p_raw_idx,
        "p_stft_x": p_stft_x,
        "p_stft_y": p_stft_y,
        "p_stft_y4": p_stft_y4,
        "has_train_idx": p_raw_idx.exists(),
    }
    if snap["has_train_idx"]:
        snap["raw_idx"] = np.load(p_raw_idx)

    n = len(snap["raw_y"])
    if not (len(snap["raw_x"]) == len(snap["raw_y4"]) == len(snap["stft_x"]) == len(snap["stft_y"]) == len(snap["stft_y4"]) == n):
        raise ValueError("Train length mismatch between raw and stft datasets.")
    return snap


def apply_fraction_train(snapshot: Dict[str, np.ndarray], indices: np.ndarray) -> None:
    np.save(snapshot["p_raw_x"], snapshot["raw_x"][indices])
    np.save(snapshot["p_raw_y"], snapshot["raw_y"][indices])
    np.save(snapshot["p_raw_y4"], snapshot["raw_y4"][indices])
    np.save(snapshot["p_stft_x"], snapshot["stft_x"][indices])
    np.save(snapshot["p_stft_y"], snapshot["stft_y"][indices])
    np.save(snapshot["p_stft_y4"], snapshot["stft_y4"][indices])
    if snapshot["has_train_idx"]:
        np.save(snapshot["p_raw_idx"], snapshot["raw_idx"][indices])


def restore_full_train(snapshot: Dict[str, np.ndarray]) -> None:
    np.save(snapshot["p_raw_x"], snapshot["raw_x"])
    np.save(snapshot["p_raw_y"], snapshot["raw_y"])
    np.save(snapshot["p_raw_y4"], snapshot["raw_y4"])
    np.save(snapshot["p_stft_x"], snapshot["stft_x"])
    np.save(snapshot["p_stft_y"], snapshot["stft_y"])
    np.save(snapshot["p_stft_y4"], snapshot["stft_y4"])
    if snapshot["has_train_idx"]:
        np.save(snapshot["p_raw_idx"], snapshot["raw_idx"])


def pick_new_file(dir_path: Path, pattern: str, before: Sequence[Path]) -> Path:
    before_set = {p.resolve() for p in before}
    after = list(dir_path.glob(pattern))
    new_files = [p for p in after if p.resolve() not in before_set]
    if not new_files:
        raise FileNotFoundError(f"No new file matched pattern '{pattern}' in {dir_path}")
    return newest_path(new_files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable used to run child scripts.")
    ap.add_argument("--out_root", type=str, default="", help="Output root directory.")
    ap.add_argument("--seed_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--fractions", type=str, default="")
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = args.python.strip() if args.python.strip() else choose_python_exe(project_root)
    dry_run = bool(args.dry_run)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"auto_seed_fraction_mfcc_darknet_cwgf_{now_str()}"
    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    tmp_ckpt_root = out_root / "tmp_ckpt"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt_root.mkdir(parents=True, exist_ok=True)

    summary_path = csv_dir / "summary_seed_fraction_metrics.csv"
    step_log_path = csv_dir / "summary_step_log.csv"

    summary_rows: List[Dict[str, str]] = []
    step_rows: List[Dict[str, str]] = []

    summary_fields = [
        "seed_name",
        "seed_value",
        "fraction",
        "train_samples",
        "train_total",
        "mfcc_mlp_4cls_macro_f1",
        "darknet_4cls_macro_f1",
        "cwgf_mfcc_darknet_4cls_macro_f1",
        "mfcc_mlp_2cls_f1",
        "darknet_2cls_f1",
        "cwgf_mfcc_darknet_2cls_f1",
        "status",
        "error",
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

    all_seeds = get_seed_list()
    seed_indices = parse_seed_indices(args.seed_indices)
    seeds = [all_seeds[i] for i in seed_indices]
    fracs = parse_fraction_list(args.fractions) if args.fractions.strip() else list(FRACTIONS)

    print("========== Auto Seed×Fraction MFCC/DarkNet/CWGF ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print(f"csv_dir    : {csv_dir}")
    print(f"stdout_dir : {stdout_dir}")
    print(f"tmp_ckpt   : {tmp_ckpt_root}")
    print(f"seed_list  : {seeds}")
    print(f"fractions  : {fracs}")
    print(f"dry_run    : {dry_run}")
    print("===========================================================")

    for seed_name, seed_value in seeds:
        seed_tag = f"{seed_name}_{seed_value}"
        print(f"\n==== Start seed: {seed_tag} ====")

        # 1) build split + stft
        build_steps = [
            (
                "build_datasets",
                [
                    python_exe,
                    "-c",
                    f"import config as cfg; cfg.SEED={seed_value}; import build_datasets as m; m.main()",
                ],
            ),
            (
                "build_stft_datasets",
                [python_exe, "build_stft_datasets.py"],
            ),
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
            dump_csv(step_log_path, step_fields, step_rows)
            if seed_failed:
                break

        if seed_failed:
            print(f"[FAIL] seed={seed_tag} build step failed, skip fractions.")
            continue

        snapshot = None
        try:
            snapshot = snapshot_train_arrays(img_size=int(getattr(cfg, "IMG_SIZE", 224)))
            n_total = int(len(snapshot["raw_y"]))
            perm = np.random.default_rng(seed_value).permutation(n_total)

            for frac in fracs:
                row = {
                    "seed_name": seed_name,
                    "seed_value": str(seed_value),
                    "fraction": f"{frac:.1f}",
                    "train_samples": "",
                    "train_total": str(n_total),
                    "mfcc_mlp_4cls_macro_f1": "",
                    "darknet_4cls_macro_f1": "",
                    "cwgf_mfcc_darknet_4cls_macro_f1": "",
                    "mfcc_mlp_2cls_f1": "",
                    "darknet_2cls_f1": "",
                    "cwgf_mfcc_darknet_2cls_f1": "",
                    "status": "ok",
                    "error": "",
                }

                n_keep = max(1, int(round(n_total * float(frac))))
                idx = perm[:n_keep]
                if not dry_run:
                    apply_fraction_train(snapshot, idx)
                row["train_samples"] = str(n_keep)
                print(f"\n-- {seed_tag} | fraction={frac:.1f} | train={n_keep}/{n_total} --")

                frac_ckpt_dir = tmp_ckpt_root / seed_tag / frac_tag(frac)
                frac_ckpt_dir.mkdir(parents=True, exist_ok=True)

                ckpt_mfcc4 = ""
                ckpt_dark4 = ""
                ckpt_mfcc2 = ""
                ckpt_dark2 = ""

                jobs = [
                    {
                        "step": "mfcc_mlp_4cls",
                        "script": "run_base_mfcc_mlp_4cls.py",
                        "args": [
                            "--data_root",
                            str(cfg.DATASET),
                            "--save_dir",
                            str(frac_ckpt_dir),
                            "--log_dir",
                            str(csv_dir),
                            "--num_workers",
                            "0",
                            "--seed",
                            str(seed_value),
                        ],
                        "csv_pat": "metrics_mlp_mfcc_4cls_*.csv",
                        "metric_cols": ["test_macro_f1"],
                        "row_key": "mfcc_mlp_4cls_macro_f1",
                        "ckpt_pat": "mlp_mfcc_4cls_best_*.pt",
                        "ckpt_key": "ckpt_mfcc4",
                    },
                    {
                        "step": "darknet_4cls",
                        "script": "run_darknet_4cls.py",
                        "args": [
                            "--data_root",
                            str(cfg.DATASET_STFT),
                            "--save_dir",
                            str(frac_ckpt_dir),
                            "--log_dir",
                            str(csv_dir),
                            "--num_workers",
                            "0",
                        ],
                        "csv_pat": "metrics_darknet19_4cls_*.csv",
                        "metric_cols": ["test_f1", "test_macro_f1"],
                        "row_key": "darknet_4cls_macro_f1",
                        "ckpt_pat": "darknet19_4cls_best_*.pth",
                        "ckpt_key": "ckpt_dark4",
                    },
                    {
                        "step": "cwgf_mfcc_darknet_4cls",
                        "script": "run_fuse_mfcc_darknet_cwgf_4cls.py",
                        "args": [
                            "--data_root_raw",
                            str(cfg.DATASET),
                            "--data_root_stft",
                            str(cfg.DATASET_STFT),
                            "--save_dir",
                            str(frac_ckpt_dir),
                            "--log_dir",
                            str(csv_dir),
                            "--save_ckpt",
                            "0",
                            "--seed",
                            str(seed_value),
                        ],
                        "csv_pat": "metrics_fuse_cwgf_mfcc_darknet_4cls_*.csv",
                        "metric_cols": ["test_macro_f1"],
                        "row_key": "cwgf_mfcc_darknet_4cls_macro_f1",
                        "ckpt_pat": "",
                        "ckpt_key": "",
                    },
                    {
                        "step": "mfcc_mlp_2cls",
                        "script": "run_base_mfcc_mlp_2cls.py",
                        "args": [
                            "--data_root",
                            str(cfg.DATASET),
                            "--save_dir",
                            str(frac_ckpt_dir),
                            "--log_dir",
                            str(csv_dir),
                            "--num_workers",
                            "0",
                            "--seed",
                            str(seed_value),
                        ],
                        "csv_pat": "metrics_mlp_mfcc_2cls_*.csv",
                        "metric_cols": ["test_f1"],
                        "row_key": "mfcc_mlp_2cls_f1",
                        "ckpt_pat": "mlp_mfcc_2cls_best_*.pt",
                        "ckpt_key": "ckpt_mfcc2",
                    },
                    {
                        "step": "darknet_2cls",
                        "script": "run_darknet_2cls.py",
                        "args": [
                            "--data_root",
                            str(cfg.DATASET_STFT),
                            "--fraction",
                            "1.0",
                            "--save_dir",
                            str(frac_ckpt_dir),
                            "--log_dir",
                            str(csv_dir),
                            "--num_workers",
                            "0",
                        ],
                        "csv_pat": "metrics_darknet19_*.csv",
                        "metric_cols": ["test_f1"],
                        "row_key": "darknet_2cls_f1",
                        "ckpt_pat": "darknet19_best_*.pth",
                        "ckpt_key": "ckpt_dark2",
                    },
                    {
                        "step": "cwgf_mfcc_darknet_2cls",
                        "script": "run_fuse_mfcc_darknet_cwgf_2cls.py",
                        "args": [
                            "--data_root_raw",
                            str(cfg.DATASET),
                            "--data_root_stft",
                            str(cfg.DATASET_STFT),
                            "--save_dir",
                            str(frac_ckpt_dir),
                            "--log_dir",
                            str(csv_dir),
                            "--save_ckpt",
                            "0",
                            "--seed",
                            str(seed_value),
                        ],
                        "csv_pat": "metrics_fuse_cwgf_mfcc_darknet_2cls_*.csv",
                        "metric_cols": ["test_f1"],
                        "row_key": "cwgf_mfcc_darknet_2cls_f1",
                        "ckpt_pat": "",
                        "ckpt_key": "",
                    },
                ]

                frac_failed = False
                for job in jobs:
                    step_name = job["step"]
                    cmd = [python_exe, job["script"]] + job["args"]

                    # inject dependent checkpoints for fusion jobs
                    if step_name == "cwgf_mfcc_darknet_4cls":
                        if (not dry_run) and (not ckpt_mfcc4 or not ckpt_dark4):
                            raise RuntimeError("Missing mfcc4/dark4 checkpoints for 4cls fusion.")
                        cmd += [
                            "--ckpt_mfcc4",
                            ckpt_mfcc4 if ckpt_mfcc4 else "DUMMY_MFCC4.pt",
                            "--ckpt_darknet4",
                            ckpt_dark4 if ckpt_dark4 else "DUMMY_DARK4.pth",
                        ]
                    if step_name == "cwgf_mfcc_darknet_2cls":
                        if (not dry_run) and (not ckpt_mfcc2 or not ckpt_dark2):
                            raise RuntimeError("Missing mfcc2/dark2 checkpoints for 2cls fusion.")
                        cmd += [
                            "--ckpt_mfcc2",
                            ckpt_mfcc2 if ckpt_mfcc2 else "DUMMY_MFCC2.pt",
                            "--ckpt_darknet2",
                            ckpt_dark2 if ckpt_dark2 else "DUMMY_DARK2.pth",
                        ]

                    t0 = time.time()
                    ts0 = dt.datetime.now().isoformat(timespec="seconds")
                    status = "ok"
                    err = ""
                    try:
                        before_csv = list(csv_dir.glob(job["csv_pat"]))
                        before_ckpt = list(frac_ckpt_dir.glob(job["ckpt_pat"])) if job["ckpt_pat"] else []

                        run_cmd(
                            cmd,
                            cwd=src_dir,
                            stdout_file=stdout_dir / f"{seed_tag}__{frac_tag(frac)}__{step_name}.log",
                            dry_run=dry_run,
                        )

                        if not dry_run:
                            new_csv = pick_new_file(csv_dir, job["csv_pat"], before_csv)
                            tagged_csv = tag_csv_name(new_csv, seed_name, seed_value, frac, step_name)
                            metric_val = parse_metric_from_csv(tagged_csv, job["metric_cols"])
                            row[job["row_key"]] = f"{metric_val:.6f}"

                            if job["ckpt_pat"]:
                                new_ckpt = pick_new_file(frac_ckpt_dir, job["ckpt_pat"], before_ckpt)
                                if job["ckpt_key"] == "ckpt_mfcc4":
                                    ckpt_mfcc4 = str(new_ckpt.resolve())
                                elif job["ckpt_key"] == "ckpt_dark4":
                                    ckpt_dark4 = str(new_ckpt.resolve())
                                elif job["ckpt_key"] == "ckpt_mfcc2":
                                    ckpt_mfcc2 = str(new_ckpt.resolve())
                                elif job["ckpt_key"] == "ckpt_dark2":
                                    ckpt_dark2 = str(new_ckpt.resolve())
                    except Exception as e:
                        status = "fail"
                        err = str(e)
                        row["status"] = "fail"
                        row["error"] = err
                        frac_failed = True

                    t1 = time.time()
                    ts1 = dt.datetime.now().isoformat(timespec="seconds")
                    step_rows.append(
                        {
                            "seed_name": seed_name,
                            "seed_value": str(seed_value),
                            "fraction": f"{frac:.1f}",
                            "step": step_name,
                            "status": status,
                            "duration_sec": f"{(t1 - t0):.2f}",
                            "error": err,
                            "start_time": ts0,
                            "end_time": ts1,
                        }
                    )
                    dump_csv(step_log_path, step_fields, step_rows)
                    if frac_failed:
                        break

                summary_rows.append(row)
                dump_csv(summary_path, summary_fields, summary_rows)

                # Keep only result files: remove temporary checkpoints for this fraction.
                if not dry_run:
                    shutil.rmtree(frac_ckpt_dir, ignore_errors=True)

                if frac_failed:
                    print(f"[WARN] failed at {seed_tag} fraction={frac:.1f} | {row['error']}")
                else:
                    print(
                        f"[DONE] {seed_tag} fr={frac:.1f} | "
                        f"4cls: mfcc={row['mfcc_mlp_4cls_macro_f1']} dark={row['darknet_4cls_macro_f1']} "
                        f"cwgf={row['cwgf_mfcc_darknet_4cls_macro_f1']} | "
                        f"2cls: mfcc={row['mfcc_mlp_2cls_f1']} dark={row['darknet_2cls_f1']} "
                        f"cwgf={row['cwgf_mfcc_darknet_2cls_f1']}"
                    )

        finally:
            if snapshot is not None:
                if not dry_run:
                    restore_full_train(snapshot)

    print("\nAll runs finished.")
    print(f"Summary: {summary_path}")
    print(f"StepLog: {step_log_path}")


if __name__ == "__main__":
    main()
