import csv
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import config as cfg


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
    with open(stdout_file, "w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        ret = subprocess.run(list(cmd), cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"return_code={ret.returncode}")


def get_seed_list() -> List[Tuple[str, int]]:
    seeds: List[Tuple[str, int]] = []
    for i in range(10):
        name = f"SEED_EXP{i}"
        if not hasattr(cfg, name):
            raise AttributeError(f"Missing {name} in config.py")
        seeds.append((name, int(getattr(cfg, name))))
    return seeds


def latest_file(dir_path: Path, pattern: str) -> Path:
    cands = list(dir_path.glob(pattern))
    if not cands:
        raise FileNotFoundError(f"No file matched: {dir_path / pattern}")
    cands.sort(key=lambda p: p.stat().st_mtime)
    return cands[-1]


def parse_test_f1_from_baseline_csv(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty csv: {csv_path}")

    best_rows = [r for r in rows if str(r.get("epoch", "")).strip().lower() == "best"]
    row = best_rows[-1] if best_rows else rows[-1]
    f1 = str(row.get("test_f1", "")).strip()
    if not f1:
        raise ValueError(f"Cannot find test_f1 in: {csv_path}")
    return f1


def parse_fusion_f1(summary_csv: Path) -> Dict[str, str]:
    with open(summary_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out = {"cwgf": "", "attention": "", "moe": ""}
    for r in rows:
        m = str(r.get("method", "")).strip().lower()
        if m in out:
            out[m] = str(r.get("fuse_f1", "")).strip()
    return out


def dump_seed_table(path: Path, rows: List[Dict[str, str]]) -> None:
    fields = [
        "seed_name",
        "seed_value",
        "resnet_f1",
        "darknet_f1",
        "cwgf_f1",
        "attention_f1",
        "moe_f1",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def dump_step_log(path: Path, rows: List[Dict[str, str]]) -> None:
    fields = [
        "seed_name",
        "seed_value",
        "step",
        "status",
        "duration_sec",
        "error",
        "start_time",
        "end_time",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = choose_python_exe(project_root)

    out_root = project_root / "runs" / f"auto_fuse2_10seeds_full_{now_str()}"
    summary_csv = out_root / "summary_seed_baseline_fusion_f1.csv"
    step_log_csv = out_root / "summary_step_log.csv"

    out_root.mkdir(parents=True, exist_ok=True)

    seed_rows: List[Dict[str, str]] = []
    step_rows: List[Dict[str, str]] = []

    seeds = get_seed_list()

    print("========== Auto 10-Seed Baseline+Fusion ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print(f"seeds      : {seeds}")
    print("===================================================")

    for seed_name, seed_value in seeds:
        seed_tag = f"{seed_name}_{seed_value}"
        print(f"\n==== Start Seed: {seed_tag} ====")

        seed_root = out_root / seed_tag
        seed_logs = seed_root / "logs"
        seed_ckpt = seed_root / "checkpoints"
        seed_stdout = seed_root / "stdout"
        seed_logs.mkdir(parents=True, exist_ok=True)
        seed_ckpt.mkdir(parents=True, exist_ok=True)
        seed_stdout.mkdir(parents=True, exist_ok=True)

        result_row = {
            "seed_name": seed_name,
            "seed_value": str(seed_value),
            "resnet_f1": "",
            "darknet_f1": "",
            "cwgf_f1": "",
            "attention_f1": "",
            "moe_f1": "",
        }

        # ---- Step 1: rebuild split + STFT ----
        steps = [
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
        for step_name, cmd in steps:
            t0 = time.time()
            ts0 = dt.datetime.now().isoformat(timespec="seconds")
            status = "ok"
            err = ""
            try:
                run_cmd(cmd, cwd=src_dir, stdout_file=seed_stdout / f"{step_name}.log")
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
            dump_step_log(step_log_csv, step_rows)

            if seed_failed:
                break

        if seed_failed:
            print(f"[FAIL] Seed {seed_tag} failed in build step.")
            seed_rows.append(result_row)
            dump_seed_table(summary_csv, seed_rows)
            continue

        # ---- Step 2: train two baseline experts ----
        baseline_jobs = [
            (
                "baseline_resnet_2cls",
                "run_base_stft_resnet_2cls.py",
                "metrics_stft_resnet18_2cls_*.csv",
                "stft_resnet18_2cls_best_*.pth",
                "resnet_f1",
            ),
            (
                "baseline_darknet_2cls",
                "run_darknet_2cls.py",
                "metrics_darknet19_*.csv",
                "darknet19_best_*.pth",
                "darknet_f1",
            ),
        ]

        ckpt_resnet = ""
        ckpt_darknet = ""

        for step_name, script_name, log_pat, ckpt_pat, key in baseline_jobs:
            t0 = time.time()
            ts0 = dt.datetime.now().isoformat(timespec="seconds")
            status = "ok"
            err = ""
            try:
                cmd = [
                    python_exe,
                    script_name,
                    "--save_dir",
                    str(seed_ckpt),
                    "--log_dir",
                    str(seed_logs),
                    "--num_workers",
                    "0",
                ]
                run_cmd(cmd, cwd=src_dir, stdout_file=seed_stdout / f"{step_name}.log")

                log_csv = latest_file(seed_logs, log_pat)
                f1 = parse_test_f1_from_baseline_csv(log_csv)
                result_row[key] = f1

                ckpt_path = latest_file(seed_ckpt, ckpt_pat)
                if key == "resnet_f1":
                    ckpt_resnet = str(ckpt_path.resolve())
                else:
                    ckpt_darknet = str(ckpt_path.resolve())
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
            dump_step_log(step_log_csv, step_rows)

            if seed_failed:
                break

        if seed_failed:
            print(f"[FAIL] Seed {seed_tag} failed in baseline training step.")
            seed_rows.append(result_row)
            dump_seed_table(summary_csv, seed_rows)
            continue

        # ---- Step 3: run 3 fusion methods auto ----
        t0 = time.time()
        ts0 = dt.datetime.now().isoformat(timespec="seconds")
        status = "ok"
        err = ""

        try:
            fuse_out = seed_root / "fusion_auto"
            cmd = [
                python_exe,
                "run_auto_fuse_2cls_3methods.py",
                "--out_root",
                str(fuse_out),
                "--ckpt_resnet",
                ckpt_resnet,
                "--ckpt_darknet",
                ckpt_darknet,
                "--save_ckpt",
                "0",
            ]
            run_cmd(cmd, cwd=src_dir, stdout_file=seed_stdout / "fusion_auto.log")

            fuse_summary = fuse_out / "csv" / "summary_auto_fuse2_3methods.csv"
            f1_map = parse_fusion_f1(fuse_summary)
            result_row["cwgf_f1"] = f1_map.get("cwgf", "")
            result_row["attention_f1"] = f1_map.get("attention", "")
            result_row["moe_f1"] = f1_map.get("moe", "")
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
                "step": "fusion_auto_3methods",
                "status": status,
                "duration_sec": f"{(t1 - t0):.2f}",
                "error": err,
                "start_time": ts0,
                "end_time": ts1,
            }
        )
        dump_step_log(step_log_csv, step_rows)

        seed_rows.append(result_row)
        dump_seed_table(summary_csv, seed_rows)

        if seed_failed:
            print(f"[FAIL] Seed {seed_tag} failed in fusion step.")
        else:
            print(
                f"[DONE] {seed_tag} | resnet_f1={result_row['resnet_f1']} "
                f"darknet_f1={result_row['darknet_f1']} "
                f"cwgf_f1={result_row['cwgf_f1']} attention_f1={result_row['attention_f1']} moe_f1={result_row['moe_f1']}"
            )

    print("\nAll seeds finished.")
    print(f"Result table : {summary_csv}")
    print(f"Step log     : {step_log_csv}")


if __name__ == "__main__":
    main()
