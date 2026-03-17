import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import config as cfg


# Note: The model order below follows the user's requested order.
# `run_base_resnet-1d_4cls.py` appears twice by design.
MODEL_SCRIPTS: List[str] = [
    "run_base_resnet-1d_2cls.py",
    "run_base_resnet-1d_4cls.py",
    "run_base_cwt_svm_2cls.py",
    "run_base_cwt_svm_4cls.py",
    "run_base_cwt_mlp_2cls.py",
    "run_base_cwt_mlp_4cls.py",
    "run_base_stft_resnet_2cls.py",
    "run_base_resnet-1d_4cls.py",
    "run_darknet_2cls.py",
    "run_darknet_4cls.py",
]


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


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


def get_seed_list(indices: Sequence[int]) -> List[Tuple[str, int]]:
    seeds: List[Tuple[str, int]] = []
    for i in indices:
        name = f"SEED_EXP{i}"
        if not hasattr(cfg, name):
            raise AttributeError(f"Missing {name} in config.py")
        seeds.append((name, int(getattr(cfg, name))))
    return seeds


def run_cmd(cmd: Sequence[str], cwd: Path, dry_run: bool = False) -> None:
    cmd_show = " ".join(cmd)
    print(f"[CMD] (cwd={cwd}) {cmd_show}")
    if dry_run:
        return
    ret = subprocess.run(list(cmd), cwd=str(cwd), check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"Command failed ({ret.returncode}): {cmd_show}")


def collect_new_csv_and_tag(
    csv_dir: Path,
    before: Sequence[Path],
    seed_tag: str,
    model_tag: str,
    dry_run: bool = False,
) -> List[str]:
    before_set = {p.resolve() for p in before}
    after = list(csv_dir.glob("*.csv"))
    new_files = [p for p in after if p.resolve() not in before_set]
    renamed: List[str] = []

    for p in sorted(new_files):
        target = p.with_name(f"{seed_tag}__{model_tag}__{p.name}")
        k = 1
        while target.exists():
            target = p.with_name(f"{seed_tag}__{model_tag}__{k:02d}__{p.name}")
            k += 1
        if not dry_run:
            p.rename(target)
        renamed.append(target.name)
    return renamed


def dump_summary(summary_path: Path, rows: List[Dict[str, str]]) -> None:
    fields = [
        "seed_name",
        "seed_value",
        "step_type",
        "step_name",
        "status",
        "duration_sec",
        "csv_outputs",
        "error",
        "start_time",
        "end_time",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def choose_python_exe(cli_python: str) -> str:
    if cli_python:
        return cli_python
    venv_py = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable used to run child scripts.")
    ap.add_argument("--seed_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_root", type=str, default="")
    ap.add_argument("--continue_on_error", type=int, default=0, help="1=continue to next step when a step fails.")
    ap.add_argument("--dry_run", type=int, default=0, help="1=print commands only.")
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    dry_run = bool(args.dry_run)
    cont = bool(args.continue_on_error)
    python_exe = choose_python_exe(args.python)

    seed_indices = parse_seed_indices(args.seed_indices)
    seed_list = get_seed_list(seed_indices)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"auto_benchmark_{now_str()}"
    csv_dir = out_root / "csv"
    ckpt_root = out_root / "checkpoints"
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(ckpt_root, exist_ok=True)

    summary_path = csv_dir / "summary_auto_benchmark.csv"
    summary_rows: List[Dict[str, str]] = []

    print("========== Auto Benchmark ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print(f"csv_dir    : {csv_dir}")
    print(f"ckpt_root  : {ckpt_root}")
    print(f"seed_list  : {seed_list}")
    print("====================================")

    for seed_name, seed_val in seed_list:
        seed_tag = f"{seed_name}_{seed_val}"
        seed_ckpt_dir = ckpt_root / seed_tag
        os.makedirs(seed_ckpt_dir, exist_ok=True)
        print(f"\n==== Start Seed: {seed_tag} ====")

        build_steps: List[Tuple[str, Sequence[str]]] = [
            ("build_datasets.py", (python_exe, "-c", f"import config as cfg; cfg.SEED={seed_val}; import build_datasets as m; m.main()")),
            ("build_cwt_datasets.py", (python_exe, "build_cwt_datasets.py")),
            ("build_stft_datasets.py", (python_exe, "build_stft_datasets.py")),
        ]

        seed_failed = False
        for step_name, cmd in build_steps:
            t0 = time.time()
            ts0 = dt.datetime.now().isoformat(timespec="seconds")
            status = "ok"
            err = ""
            csv_outputs = ""
            try:
                run_cmd(cmd, cwd=src_dir, dry_run=dry_run)
            except Exception as e:
                status = "fail"
                err = str(e)
                seed_failed = True
            t1 = time.time()
            ts1 = dt.datetime.now().isoformat(timespec="seconds")

            summary_rows.append(
                {
                    "seed_name": seed_name,
                    "seed_value": str(seed_val),
                    "step_type": "build",
                    "step_name": step_name,
                    "status": status,
                    "duration_sec": f"{(t1 - t0):.2f}",
                    "csv_outputs": csv_outputs,
                    "error": err,
                    "start_time": ts0,
                    "end_time": ts1,
                }
            )
            dump_summary(summary_path, summary_rows)

            if status == "fail" and not cont:
                raise RuntimeError(f"[{seed_tag}] build step failed: {step_name} | {err}")
            if status == "fail" and cont:
                break

        if seed_failed and cont:
            print(f"[WARN] Skip model runs for seed {seed_tag} due to build failure.")
            continue

        for i, script_name in enumerate(MODEL_SCRIPTS, start=1):
            script_path = src_dir / script_name
            if not script_path.exists():
                msg = f"script not found: {script_path}"
                if not cont:
                    raise FileNotFoundError(msg)
                print(f"[WARN] {msg}")
                summary_rows.append(
                    {
                        "seed_name": seed_name,
                        "seed_value": str(seed_val),
                        "step_type": "model",
                        "step_name": script_name,
                        "status": "fail",
                        "duration_sec": "0.00",
                        "csv_outputs": "",
                        "error": msg,
                        "start_time": dt.datetime.now().isoformat(timespec="seconds"),
                        "end_time": dt.datetime.now().isoformat(timespec="seconds"),
                    }
                )
                dump_summary(summary_path, summary_rows)
                continue

            t0 = time.time()
            ts0 = dt.datetime.now().isoformat(timespec="seconds")
            status = "ok"
            err = ""
            csv_outputs = ""
            model_tag = f"m{i:02d}_{script_path.stem}"
            before_csv = list(csv_dir.glob("*.csv"))

            cmd = [
                python_exe,
                script_name,
                "--save_dir",
                str(seed_ckpt_dir),
                "--log_dir",
                str(csv_dir),
            ]

            try:
                run_cmd(cmd, cwd=src_dir, dry_run=dry_run)
                new_csvs = collect_new_csv_and_tag(
                    csv_dir=csv_dir,
                    before=before_csv,
                    seed_tag=seed_tag,
                    model_tag=model_tag,
                    dry_run=dry_run,
                )
                csv_outputs = ";".join(new_csvs)
            except Exception as e:
                status = "fail"
                err = str(e)

            t1 = time.time()
            ts1 = dt.datetime.now().isoformat(timespec="seconds")
            summary_rows.append(
                {
                    "seed_name": seed_name,
                    "seed_value": str(seed_val),
                    "step_type": "model",
                    "step_name": script_name,
                    "status": status,
                    "duration_sec": f"{(t1 - t0):.2f}",
                    "csv_outputs": csv_outputs,
                    "error": err,
                    "start_time": ts0,
                    "end_time": ts1,
                }
            )
            dump_summary(summary_path, summary_rows)

            if status == "fail" and not cont:
                raise RuntimeError(f"[{seed_tag}] model step failed: {script_name} | {err}")

        print(f"==== Finished Seed: {seed_tag} ====")

    print("\nAll requested runs finished.")
    print(f"CSV directory : {csv_dir}")
    print(f"Weights dir   : {ckpt_root}")
    print(f"Summary CSV   : {summary_path}")


if __name__ == "__main__":
    main()
