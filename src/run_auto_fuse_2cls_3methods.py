import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import config as cfg


METHOD_SCRIPTS = [
    ("cwgf", "run_fuse_2cls_cwgf_network.py"),
    ("attention", "run_fuse_2cls_attention_network.py"),
    ("moe", "run_fuse_2cls_moe_network.py"),
]


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def choose_python_exe(cli_python: str, project_root: Path) -> str:
    if cli_python.strip():
        return cli_python
    py_venv_unix = project_root / ".venv" / "bin" / "python"
    if py_venv_unix.exists():
        return str(py_venv_unix)
    py_venv_win = project_root / ".venv" / "Scripts" / "python.exe"
    if py_venv_win.exists():
        return str(py_venv_win)
    return sys.executable


def collect_new_csv(
    csv_dir: Path,
    before: Sequence[Path],
    method: str,
    dry_run: bool,
) -> List[Path]:
    before_set = {p.resolve() for p in before}
    after = sorted(csv_dir.glob("metrics_fuse2_*_resnet_darknet_*.csv"), key=lambda p: p.stat().st_mtime)
    new_files = [p for p in after if p.resolve() not in before_set]

    out: List[Path] = []
    for p in new_files:
        target = p.with_name(f"{method}__{p.name}")
        k = 1
        while target.exists():
            target = p.with_name(f"{method}__{k:02d}__{p.name}")
            k += 1
        if not dry_run:
            p.rename(target)
        out.append(target)
    return out


def parse_metrics_csv(path: Path) -> Dict[str, str]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.reader(f):
            rows.append(r)

    if len(rows) < 6:
        raise ValueError(f"Unexpected csv format: {path}")

    head = rows[0]
    val = rows[1]
    if len(head) != len(val):
        raise ValueError(f"Header/value length mismatch: {path}")

    fused = {head[i]: val[i] for i in range(len(head))}

    # Expert rows are after a blank line + expert header.
    # Expected layout:
    # row0 fused header
    # row1 fused value
    # row2 blank
    # row3 expert header
    # row4 resnet row
    # row5 darknet row
    exp_header = rows[3]
    exp_res = rows[4]
    exp_dark = rows[5]
    if len(exp_header) != len(exp_res) or len(exp_header) != len(exp_dark):
        raise ValueError(f"Expert header/value length mismatch: {path}")

    res_map = {exp_header[i]: exp_res[i] for i in range(len(exp_header))}
    dark_map = {exp_header[i]: exp_dark[i] for i in range(len(exp_header))}

    out = {
        "method": fused.get("method", ""),
        "tau": fused.get("tau", ""),
        "fuse_acc": fused.get("fuse_acc", ""),
        "fuse_f1": fused.get("fuse_f1", ""),
        "fuse_prec": fused.get("fuse_prec", ""),
        "fuse_rec": fused.get("fuse_rec", ""),
        "fuse_bal_acc": fused.get("fuse_bal_acc", ""),
        "best_val_f1": fused.get("best_val_f1", ""),
        "resnet_acc": res_map.get("acc", ""),
        "resnet_f1": res_map.get("f1", ""),
        "darknet_acc": dark_map.get("acc", ""),
        "darknet_f1": dark_map.get("f1", ""),
    }
    return out


def dump_summary(path: Path, rows: List[Dict[str, str]]) -> None:
    fields = [
        "method",
        "script",
        "status",
        "duration_sec",
        "metrics_csv",
        "tau",
        "fuse_acc",
        "fuse_f1",
        "fuse_prec",
        "fuse_rec",
        "fuse_bal_acc",
        "best_val_f1",
        "resnet_acc",
        "resnet_f1",
        "darknet_acc",
        "darknet_f1",
        "error",
        "start_time",
        "end_time",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_cmd(cmd: Sequence[str], cwd: Path, stdout_file: Path, dry_run: bool) -> None:
    print(f"[CMD] (cwd={cwd}) {' '.join(cmd)}")
    if dry_run:
        return

    with open(stdout_file, "w", encoding="utf-8") as f:
        f.write("[CMD] " + " ".join(cmd) + "\n\n")
        ret = subprocess.run(list(cmd), cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, check=False)
    if ret.returncode != 0:
        raise RuntimeError(f"return_code={ret.returncode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable for child scripts")
    ap.add_argument("--out_root", type=str, default="")

    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt_resnet", type=str, default="")
    ap.add_argument("--ckpt_darknet", type=str, default="")

    ap.add_argument("--fusion_hid", type=int, default=16)
    ap.add_argument("--fusion_drop", type=float, default=0.1)
    ap.add_argument("--fusion_epochs", type=int, default=200)
    ap.add_argument("--fusion_lr", type=float, default=1e-4)
    ap.add_argument("--fusion_wd", type=float, default=1e-3)
    ap.add_argument("--lambda_ent", type=float, default=0.1)
    ap.add_argument("--lambda_center", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=20)

    ap.add_argument("--thr_lo", type=float, default=0.20)
    ap.add_argument("--thr_hi", type=float, default=0.80)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--thr_objective", type=str, default="f1", choices=["f1", "acc"])

    ap.add_argument("--pos_pow", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=getattr(cfg, "SEED", 2025))

    ap.add_argument("--save_ckpt", type=int, default=0)
    ap.add_argument("--continue_on_error", type=int, default=0)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = choose_python_exe(args.python, project_root)
    dry_run = bool(args.dry_run)
    cont = bool(args.continue_on_error)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"auto_fuse2_3methods_{now_str()}"

    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    ckpt_dir = out_root / "checkpoints"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.save_ckpt):
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = csv_dir / "summary_auto_fuse2_3methods.csv"
    rows: List[Dict[str, str]] = []

    print("========== Auto Fuse 2cls (3 methods) ==========")
    print(f"python_exe  : {python_exe}")
    print(f"src_dir     : {src_dir}")
    print(f"out_root    : {out_root}")
    print(f"csv_dir     : {csv_dir}")
    print(f"stdout_dir  : {stdout_dir}")
    print(f"ckpt_dir    : {ckpt_dir if bool(args.save_ckpt) else '(disabled)'}")
    print("methods     : cwgf, attention, moe")
    print("=================================================")

    common_args = [
        "--data_root", str(args.data_root),
        "--img_size", str(args.img_size),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--fusion_hid", str(args.fusion_hid),
        "--fusion_drop", str(args.fusion_drop),
        "--fusion_epochs", str(args.fusion_epochs),
        "--fusion_lr", str(args.fusion_lr),
        "--fusion_wd", str(args.fusion_wd),
        "--lambda_ent", str(args.lambda_ent),
        "--lambda_center", str(args.lambda_center),
        "--grad_clip", str(args.grad_clip),
        "--eval_every", str(args.eval_every),
        "--thr_lo", str(args.thr_lo),
        "--thr_hi", str(args.thr_hi),
        "--thr_step", str(args.thr_step),
        "--thr_objective", str(args.thr_objective),
        "--pos_pow", str(args.pos_pow),
        "--seed", str(args.seed),
        "--save_ckpt", str(args.save_ckpt),
        "--save_dir", str(ckpt_dir),
        "--log_dir", str(csv_dir),
    ]
    if args.ckpt_resnet.strip():
        common_args += ["--ckpt_resnet", args.ckpt_resnet]
    if args.ckpt_darknet.strip():
        common_args += ["--ckpt_darknet", args.ckpt_darknet]

    for method, script in METHOD_SCRIPTS:
        script_path = src_dir / script
        t0 = time.time()
        ts0 = dt.datetime.now().isoformat(timespec="seconds")
        status = "ok"
        err = ""
        metrics_csv = ""
        parsed: Dict[str, str] = {}

        if not script_path.exists():
            status = "fail"
            err = f"script not found: {script_path}"
        else:
            before_csv = list(csv_dir.glob("metrics_fuse2_*_resnet_darknet_*.csv"))
            stdout_file = stdout_dir / f"{method}.log"
            cmd = [python_exe, script] + common_args
            try:
                run_cmd(cmd, cwd=src_dir, stdout_file=stdout_file, dry_run=dry_run)
                if not dry_run:
                    new_csvs = collect_new_csv(csv_dir, before_csv, method=method, dry_run=dry_run)
                    if not new_csvs:
                        status = "fail"
                        err = "no_new_metrics_csv"
                    else:
                        metrics_csv = str(new_csvs[-1].name)
                        parsed = parse_metrics_csv(new_csvs[-1])
            except Exception as e:
                status = "fail"
                err = str(e)

        t1 = time.time()
        ts1 = dt.datetime.now().isoformat(timespec="seconds")

        row = {
            "method": method,
            "script": script,
            "status": status,
            "duration_sec": f"{(t1 - t0):.2f}",
            "metrics_csv": metrics_csv,
            "tau": parsed.get("tau", ""),
            "fuse_acc": parsed.get("fuse_acc", ""),
            "fuse_f1": parsed.get("fuse_f1", ""),
            "fuse_prec": parsed.get("fuse_prec", ""),
            "fuse_rec": parsed.get("fuse_rec", ""),
            "fuse_bal_acc": parsed.get("fuse_bal_acc", ""),
            "best_val_f1": parsed.get("best_val_f1", ""),
            "resnet_acc": parsed.get("resnet_acc", ""),
            "resnet_f1": parsed.get("resnet_f1", ""),
            "darknet_acc": parsed.get("darknet_acc", ""),
            "darknet_f1": parsed.get("darknet_f1", ""),
            "error": err,
            "start_time": ts0,
            "end_time": ts1,
        }
        rows.append(row)
        dump_summary(summary_csv, rows)

        if status == "ok":
            print(
                f"[DONE] {method:9s} | f1={row['fuse_f1']} rec={row['fuse_rec']} "
                f"tau={row['tau']} | csv={metrics_csv}"
            )
        else:
            print(f"[FAIL] {method:9s} | {err}")
            if not cont:
                raise RuntimeError(f"Method failed: {method} | {err}")

    print("\nAll requested fusion runs finished.")
    print(f"Summary CSV : {summary_csv}")
    print(f"CSV dir     : {csv_dir}")
    print(f"STDOUT logs : {stdout_dir}")


if __name__ == "__main__":
    main()
