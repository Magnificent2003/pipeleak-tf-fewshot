import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def choose_python_exe(project_root: Path) -> str:
    py_venv_unix = project_root / ".venv" / "bin" / "python"
    if py_venv_unix.exists():
        return str(py_venv_unix)
    py_venv_win = project_root / ".venv" / "Scripts" / "python.exe"
    if py_venv_win.exists():
        return str(py_venv_win)
    return sys.executable


def run_step(name: str, cmd: List[str], cwd: Path) -> None:
    t0 = dt.datetime.now()
    print(f"\n========== START: {name} ==========")
    print("[CMD]", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=str(cwd), check=False)
    t1 = dt.datetime.now()
    dt_sec = (t1 - t0).total_seconds()
    if ret.returncode != 0:
        print(f"========== FAIL : {name} | code={ret.returncode} | {dt_sec:.1f}s ==========")
        raise RuntimeError(f"Step failed: {name} | return_code={ret.returncode}")
    print(f"========== DONE : {name} | {dt_sec:.1f}s ==========")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable for child scripts.")
    ap.add_argument("--seed_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = args.python.strip() if args.python.strip() else choose_python_exe(project_root)

    jobs: List[Tuple[str, str]] = [
        ("auto#1 mfccsvm+mfccmlp+cwgf", "run_auto_seed_fraction06_mfccsvm_mfccmlp_cwgf.py"),
        ("auto#2 resnet2+cwtmlp2+cwgf", "run_auto_seed_fraction06_resnet2_cwtmlp2_cwgf.py"),
    ]

    print("========== Two-Stage Auto Runner (fraction=0.6) ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"seed_indices: {args.seed_indices}")
    print(f"dry_run    : {bool(args.dry_run)}")
    print("===========================================================")

    for name, script in jobs:
        cmd = [
            python_exe,
            script,
            "--seed_indices",
            args.seed_indices,
            "--dry_run",
            str(int(args.dry_run)),
        ]
        run_step(name, cmd, cwd=src_dir)

    print("\nAll done. Two auto scripts finished in sequence.")


if __name__ == "__main__":
    main()
