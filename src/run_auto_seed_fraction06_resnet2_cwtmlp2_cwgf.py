import argparse
import csv
import datetime as dt
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

import config as cfg


TRAIN_FRACTION = 0.6


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


def dump_csv(path: Path, fields: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def tag_csv_name(csv_path: Path, seed_name: str, seed_value: int, model_tag: str) -> Path:
    prefix = f"{seed_name}_{seed_value}__fr_0p6__{model_tag}__"
    target = csv_path.with_name(prefix + csv_path.name)
    k = 1
    while target.exists():
        target = csv_path.with_name(prefix + f"{k:02d}__" + csv_path.name)
        k += 1
    csv_path.rename(target)
    return target


def read_best_row(csv_path: Path) -> Dict[str, str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty csv: {csv_path}")
    best_rows = [r for r in rows if str(r.get("epoch", "")).strip().lower() == "best"]
    return best_rows[-1] if best_rows else rows[-1]


def read_single_row_csv(csv_path: Path) -> Dict[str, str]:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty csv: {csv_path}")
    return rows[-1]


def parse_metric_from_row(row: Dict[str, str], cols: Sequence[str]) -> float:
    for c in cols:
        v = str(row.get(c, "")).strip()
        if v != "":
            return float(v)
    raise ValueError(f"Cannot parse any metric in columns={cols} from row keys={list(row.keys())}")


def apply_train_fraction_raw(raw_root: Path, fraction: float, seed: int) -> Tuple[int, int]:
    p_x = raw_root / "X_train.npy"
    p_y = raw_root / "y_train.npy"
    p_y4 = raw_root / "y4_train.npy"
    p_idx = raw_root / "train_idx.npy"

    required = [p_x, p_y, p_y4]
    miss = [str(p) for p in required if not p.exists()]
    if miss:
        raise FileNotFoundError("Missing raw train files:\n" + "\n".join(miss))

    x = np.load(p_x)
    y = np.load(p_y).astype(np.int64)
    y4 = np.load(p_y4).astype(np.int64)
    idx_raw = np.load(p_idx) if p_idx.exists() else None

    rng = np.random.default_rng(int(seed))
    picked = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        keep = max(1, int(round(len(idx) * float(fraction))))
        if len(idx) < keep:
            raise ValueError(f"class {cls} has only {len(idx)} samples, need {keep}")
        picked.append(rng.choice(idx, size=keep, replace=False))
    sel = np.concatenate(picked, axis=0).astype(np.int64)
    rng.shuffle(sel)

    np.save(p_x, x[sel])
    np.save(p_y, y[sel])
    np.save(p_y4, y4[sel])
    if idx_raw is not None:
        np.save(p_idx, idx_raw[sel])

    return int(len(sel)), int(len(y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable used to run child scripts.")
    ap.add_argument("--out_root", type=str, default="", help="Output root directory.")
    ap.add_argument("--seed_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = args.python.strip() if args.python.strip() else choose_python_exe(project_root)
    dry_run = bool(args.dry_run)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"auto_seed_fraction06_resnet2_cwtmlp2_cwgf_{now_str()}"

    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    ckpt_root = out_root / "checkpoints"
    data_raw_root = out_root / "data_raw"
    data_stft_root = out_root / "data_stft"
    data_cwt_root = out_root / "data_cwt"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)
    data_raw_root.mkdir(parents=True, exist_ok=True)
    data_stft_root.mkdir(parents=True, exist_ok=True)
    data_cwt_root.mkdir(parents=True, exist_ok=True)

    summary_csv = csv_dir / "summary_seed06_resnet2_cwtmlp2_cwgf_f1.csv"
    step_csv = csv_dir / "step_log.csv"

    summary_fields = [
        "seed_name",
        "seed_value",
        "train_fraction",
        "train_samples",
        "train_total",
        "resnet18_2cls_f1",
        "cwt_mlp_2cls_f1",
        "cwgf_status",
        "cwgf_f1",
        "resnet_csv",
        "cwt_mlp_csv",
        "cwgf_csv",
        "status",
        "error",
    ]
    step_fields = [
        "seed_name",
        "seed_value",
        "step",
        "status",
        "duration_sec",
        "error",
        "start_time",
        "end_time",
    ]
    summary_rows: List[Dict[str, str]] = []
    step_rows: List[Dict[str, str]] = []

    all_seeds = get_seed_list()
    seed_indices = parse_seed_indices(args.seed_indices)
    seed_list = [all_seeds[i] for i in seed_indices]

    print("========== Auto Seed (fraction=0.6): ResNet2 + CWT-MLP2 + CWGF ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print(f"seed_list  : {seed_list}")
    print(f"fraction   : {TRAIN_FRACTION}")
    print(f"dry_run    : {dry_run}")
    print("=========================================================================")

    for seed_name, seed_value in seed_list:
        seed_tag = f"{seed_name}_{seed_value}"
        print(f"\n==== Start seed: {seed_tag} ====")
        row = {
            "seed_name": seed_name,
            "seed_value": str(seed_value),
            "train_fraction": f"{TRAIN_FRACTION:.1f}",
            "train_samples": "",
            "train_total": "",
            "resnet18_2cls_f1": "",
            "cwt_mlp_2cls_f1": "",
            "cwgf_status": "",
            "cwgf_f1": "",
            "resnet_csv": "",
            "cwt_mlp_csv": "",
            "cwgf_csv": "",
            "status": "ok",
            "error": "",
        }

        seed_raw = data_raw_root / seed_tag
        seed_stft = data_stft_root / seed_tag
        seed_cwt = data_cwt_root / seed_tag
        seed_ckpt = ckpt_root / seed_tag
        seed_raw.mkdir(parents=True, exist_ok=True)
        seed_stft.mkdir(parents=True, exist_ok=True)
        seed_cwt.mkdir(parents=True, exist_ok=True)
        seed_ckpt.mkdir(parents=True, exist_ok=True)

        build_steps = [
            (
                "build_datasets_seed",
                [
                    python_exe,
                    "-c",
                    (
                        "import config as cfg; "
                        f"cfg.SEED={seed_value}; "
                        f"cfg.DATASET=r'{seed_raw.as_posix()}'; "
                        "import build_datasets as m; m.main()"
                    ),
                ],
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
                    "step": step_name,
                    "status": status,
                    "duration_sec": f"{(t1 - t0):.2f}",
                    "error": err,
                    "start_time": ts0,
                    "end_time": ts1,
                }
            )
            dump_csv(step_csv, step_fields, step_rows)
            if seed_failed:
                break

        if seed_failed:
            row["status"] = "fail"
            row["error"] = "build_datasets failed"
            summary_rows.append(row)
            dump_csv(summary_csv, summary_fields, summary_rows)
            continue

        try:
            if not dry_run:
                n_keep, n_total = apply_train_fraction_raw(seed_raw, TRAIN_FRACTION, seed_value)
                row["train_samples"] = str(n_keep)
                row["train_total"] = str(n_total)
                print(f"[Fraction] kept={n_keep}/{n_total} on {seed_tag}")
            else:
                row["train_samples"] = "DRY_RUN"
                row["train_total"] = "DRY_RUN"
        except Exception as e:
            row["status"] = "fail"
            row["error"] = f"apply_fraction failed: {e}"
            summary_rows.append(row)
            dump_csv(summary_csv, summary_fields, summary_rows)
            continue

        # build CWT
        t0 = time.time()
        ts0 = dt.datetime.now().isoformat(timespec="seconds")
        status = "ok"
        err = ""
        try:
            cmd = [
                python_exe,
                "-c",
                (
                    "import config as cfg; "
                    f"cfg.DATASET=r'{seed_raw.as_posix()}'; "
                    f"cfg.DATASET_CWT=r'{seed_cwt.as_posix()}'; "
                    "import build_cwt_datasets as m; m.main()"
                ),
            ]
            run_cmd(cmd, cwd=src_dir, stdout_file=stdout_dir / f"{seed_tag}__build_cwt.log", dry_run=dry_run)
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
                "step": "build_cwt_datasets",
                "status": status,
                "duration_sec": f"{(t1 - t0):.2f}",
                "error": err,
                "start_time": ts0,
                "end_time": ts1,
            }
        )
        dump_csv(step_csv, step_fields, step_rows)
        if seed_failed:
            row["status"] = "fail"
            row["error"] = f"build_cwt failed: {err}"
            summary_rows.append(row)
            dump_csv(summary_csv, summary_fields, summary_rows)
            continue

        # build STFT
        t0 = time.time()
        ts0 = dt.datetime.now().isoformat(timespec="seconds")
        status = "ok"
        err = ""
        try:
            cmd = [
                python_exe,
                "-c",
                (
                    "import config as cfg; "
                    f"cfg.DATASET=r'{seed_raw.as_posix()}'; "
                    f"cfg.DATASET_STFT=r'{seed_stft.as_posix()}'; "
                    "cfg.SAVE_PNG=False; "
                    "import build_stft_datasets as m; m.main()"
                ),
            ]
            run_cmd(cmd, cwd=src_dir, stdout_file=stdout_dir / f"{seed_tag}__build_stft.log", dry_run=dry_run)
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
                "step": "build_stft_datasets",
                "status": status,
                "duration_sec": f"{(t1 - t0):.2f}",
                "error": err,
                "start_time": ts0,
                "end_time": ts1,
            }
        )
        dump_csv(step_csv, step_fields, step_rows)
        if seed_failed:
            row["status"] = "fail"
            row["error"] = f"build_stft failed: {err}"
            summary_rows.append(row)
            dump_csv(summary_csv, summary_fields, summary_rows)
            continue

        jobs = [
            {
                "step": "resnet2",
                "script": "run_base_stft_resnet_2cls.py",
                "args": [
                    "--data_root",
                    str(seed_stft),
                    "--save_dir",
                    str(seed_ckpt),
                    "--log_dir",
                    str(csv_dir),
                    "--num_workers",
                    "0",
                ],
                "csv_pat": "metrics_stft_resnet18_2cls_*.csv",
                "ckpt_pat": "stft_resnet18_2cls_best_*.pth",
                "tag": "resnet2",
            },
            {
                "step": "cwt_mlp2",
                "script": "run_base_cwt_mlp_2cls.py",
                "args": [
                    "--data_root",
                    str(seed_cwt),
                    "--save_dir",
                    str(seed_ckpt),
                    "--log_dir",
                    str(csv_dir),
                    "--num_workers",
                    "0",
                    "--seed",
                    str(seed_value),
                ],
                "csv_pat": "metrics_mlp_cwt_2cls_*.csv",
                "ckpt_pat": "mlp_cwt_2cls_best_*.pt",
                "tag": "cwt_mlp2",
            },
        ]

        resnet_ckpt = ""
        cwt_ckpt = ""
        resnet_csv_tagged = None
        cwt_csv_tagged = None

        for job in jobs:
            t0 = time.time()
            ts0 = dt.datetime.now().isoformat(timespec="seconds")
            status = "ok"
            err = ""
            try:
                before_csv = list(csv_dir.glob(job["csv_pat"]))
                before_ckpt = list(seed_ckpt.glob(job["ckpt_pat"]))
                cmd = [python_exe, job["script"]] + job["args"]
                run_cmd(
                    cmd,
                    cwd=src_dir,
                    stdout_file=stdout_dir / f"{seed_tag}__{job['step']}.log",
                    dry_run=dry_run,
                )
                if not dry_run:
                    new_csv = pick_new_file(csv_dir, job["csv_pat"], before_csv)
                    tagged_csv = tag_csv_name(new_csv, seed_name, seed_value, job["tag"])
                    new_ckpt = pick_new_file(seed_ckpt, job["ckpt_pat"], before_ckpt)
                    if job["step"] == "resnet2":
                        resnet_ckpt = str(new_ckpt.resolve())
                        resnet_csv_tagged = tagged_csv
                    elif job["step"] == "cwt_mlp2":
                        cwt_ckpt = str(new_ckpt.resolve())
                        cwt_csv_tagged = tagged_csv
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
                    "step": job["step"],
                    "status": status,
                    "duration_sec": f"{(t1 - t0):.2f}",
                    "error": err,
                    "start_time": ts0,
                    "end_time": ts1,
                }
            )
            dump_csv(step_csv, step_fields, step_rows)
            if seed_failed:
                break

        if seed_failed:
            row["status"] = "fail"
            row["error"] = "backbone training failed"
            summary_rows.append(row)
            dump_csv(summary_csv, summary_fields, summary_rows)
            continue

        # run fusion
        t0 = time.time()
        ts0 = dt.datetime.now().isoformat(timespec="seconds")
        status = "ok"
        err = ""
        fuse_csv_tagged = None
        try:
            before_csv = list(csv_dir.glob("metrics_fuse_cwgf_resnet2_cwtmlp2_2cls_*.csv"))
            cmd = [
                python_exe,
                "run_fuse_cwgf_resnet2_cwtmlp2.py",
                "--data_root_stft",
                str(seed_stft),
                "--data_root_cwt",
                str(seed_cwt),
                "--save_dir",
                str(seed_ckpt),
                "--log_dir",
                str(csv_dir),
                "--save_ckpt",
                "0",
                "--num_workers",
                "0",
                "--seed",
                str(seed_value),
                "--ckpt_resnet2",
                str(resnet_ckpt),
                "--ckpt_cwt_mlp2",
                str(cwt_ckpt),
            ]
            run_cmd(cmd, cwd=src_dir, stdout_file=stdout_dir / f"{seed_tag}__fuse_cwgf.log", dry_run=dry_run)
            if not dry_run:
                new_csv = pick_new_file(csv_dir, "metrics_fuse_cwgf_resnet2_cwtmlp2_2cls_*.csv", before_csv)
                fuse_csv_tagged = tag_csv_name(new_csv, seed_name, seed_value, "fuse_cwgf_resnet2_cwtmlp2")
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
                "step": "fuse_cwgf",
                "status": status,
                "duration_sec": f"{(t1 - t0):.2f}",
                "error": err,
                "start_time": ts0,
                "end_time": ts1,
            }
        )
        dump_csv(step_csv, step_fields, step_rows)
        if seed_failed:
            row["status"] = "fail"
            row["error"] = f"fusion failed: {err}"
            summary_rows.append(row)
            dump_csv(summary_csv, summary_fields, summary_rows)
            continue

        # parse metrics
        try:
            if not dry_run:
                resnet_best = read_best_row(resnet_csv_tagged)
                cwt_best = read_best_row(cwt_csv_tagged)
                fuse_last = read_single_row_csv(fuse_csv_tagged)

                row["resnet18_2cls_f1"] = f"{parse_metric_from_row(resnet_best, ['test_f1']):.6f}"
                row["cwt_mlp_2cls_f1"] = f"{parse_metric_from_row(cwt_best, ['test_f1']):.6f}"
                row["cwgf_status"] = str(fuse_last.get("degeneracy", "")).strip()
                row["cwgf_f1"] = f"{float(str(fuse_last.get('test_fuse_f1', '')).strip()):.6f}"
                row["resnet_csv"] = resnet_csv_tagged.name
                row["cwt_mlp_csv"] = cwt_csv_tagged.name
                row["cwgf_csv"] = fuse_csv_tagged.name
            else:
                row["resnet18_2cls_f1"] = "DRY_RUN"
                row["cwt_mlp_2cls_f1"] = "DRY_RUN"
                row["cwgf_status"] = "DRY_RUN"
                row["cwgf_f1"] = "DRY_RUN"
        except Exception as e:
            row["status"] = "fail"
            row["error"] = f"parse metrics failed: {e}"

        summary_rows.append(row)
        dump_csv(summary_csv, summary_fields, summary_rows)

    print("\nAll done.")
    print(f"[Summary] {summary_csv}")
    print(f"[StepLog] {step_csv}")


if __name__ == "__main__":
    main()
