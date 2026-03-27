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


FRACTIONS: List[float] = [1.0, 0.8, 0.6, 0.4]


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


def dump_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


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


def get_seed_list() -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for i in range(10):
        name = f"SEED_EXP{i}"
        if not hasattr(cfg, name):
            raise AttributeError(f"Missing {name} in config.py")
        out.append((name, int(getattr(cfg, name))))
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
        raise FileNotFoundError(f"No new file matched pattern '{pattern}' in {dir_path}")
    return newest_path(new_files)


def parse_metric_from_csv(csv_path: Path, metric_cols: Sequence[str]) -> float:
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty csv: {csv_path}")

    best_rows = [r for r in rows if str(r.get("epoch", "")).strip().lower() == "best"]
    row = best_rows[-1] if best_rows else rows[-1]
    for c in metric_cols:
        v = str(row.get(c, "")).strip()
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


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


class MAMLTrainSnapshot:
    """
    Only train split is fractioned:
      - y4_train.npy
      - png_train/*
    val/test are untouched.
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.p_y4_train = data_root / "y4_train.npy"
        self.p_png_train = data_root / "png_train"
        self.p_png_backup = data_root / "png_train__full_backup_auto_maml"

        if not self.p_y4_train.exists():
            raise FileNotFoundError(f"Missing {self.p_y4_train}")
        if not self.p_png_train.exists():
            raise FileNotFoundError(f"Missing {self.p_png_train}")
        if self.p_png_backup.exists():
            raise FileExistsError(f"Temporary backup dir already exists: {self.p_png_backup}")

        self.y4_full = np.load(self.p_y4_train).astype(np.int64)
        self.n_total = int(len(self.y4_full))
        os.replace(self.p_png_train, self.p_png_backup)

    def apply_fraction(self, indices: np.ndarray) -> int:
        n_keep = int(len(indices))
        _safe_rmtree(self.p_png_train)
        self.p_png_train.mkdir(parents=True, exist_ok=True)

        y4_sub = self.y4_full[indices].astype(np.int64)
        np.save(self.p_y4_train, y4_sub)

        for new_i, old_i in enumerate(indices.tolist()):
            src = self.p_png_backup / f"train_{int(old_i):06d}.png"
            if not src.exists():
                raise FileNotFoundError(f"Missing source image: {src}")
            dst = self.p_png_train / f"train_{new_i:06d}.png"
            _link_or_copy(src, dst)
        return n_keep

    def restore(self) -> None:
        np.save(self.p_y4_train, self.y4_full)
        _safe_rmtree(self.p_png_train)
        if self.p_png_backup.exists():
            os.replace(self.p_png_backup, self.p_png_train)


def aggregate_mean_sd_by_fraction(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    by_frac: Dict[str, List[float]] = {}
    for r in rows:
        if r.get("status", "") != "ok":
            continue
        frac = r.get("fraction", "")
        val = r.get("maml_4cls_macro_f1", "")
        if frac == "" or val == "":
            continue
        by_frac.setdefault(frac, []).append(float(val))

    out: List[Dict[str, str]] = []
    for frac in sorted(by_frac.keys(), key=lambda x: float(x)):
        arr = np.array(by_frac[frac], dtype=np.float64)
        mean = float(arr.mean()) if arr.size > 0 else float("nan")
        sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        out.append(
            {
                "fraction": frac,
                "n": str(int(arr.size)),
                "macro_f1_mean": f"{mean:.6f}",
                "macro_f1_sd": f"{sd:.6f}",
                "macro_f1_mean_pm_sd": f"{mean:.6f} ± {sd:.6f}",
            }
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", type=str, default="", help="Python executable used to run child scripts.")
    ap.add_argument("--out_root", type=str, default="", help="Output root directory.")
    ap.add_argument("--seed_indices", type=str, default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--fractions", type=str, default="")
    ap.add_argument("--maml_epochs", type=int, default=cfg.MAML_EPOCHS)
    ap.add_argument("--maml_train_episodes", type=int, default=cfg.MAML_TRAIN_EPISODES)
    ap.add_argument("--maml_val_episodes", type=int, default=cfg.MAML_VAL_EPISODES)
    ap.add_argument("--maml_test_episodes", type=int, default=cfg.MAML_TEST_EPISODES)
    ap.add_argument("--maml_meta_batch", type=int, default=cfg.MAML_META_BATCH)
    ap.add_argument("--maml_num_workers", type=int, default=0)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    python_exe = args.python.strip() if args.python.strip() else choose_python_exe(project_root)
    dry_run = bool(args.dry_run)

    if args.out_root.strip():
        out_root = Path(args.out_root).resolve()
    else:
        out_root = project_root / "runs" / f"auto_seed_fraction_maml4_{now_str()}"
    csv_dir = out_root / "csv"
    stdout_dir = out_root / "stdout"
    tmp_ckpt_root = out_root / "tmp_ckpt"
    csv_dir.mkdir(parents=True, exist_ok=True)
    stdout_dir.mkdir(parents=True, exist_ok=True)
    tmp_ckpt_root.mkdir(parents=True, exist_ok=True)

    summary_path = csv_dir / "summary_seed_fraction_maml4_macro_f1.csv"
    mean_sd_path = csv_dir / "summary_fraction_maml4_macro_f1_mean_sd.csv"
    step_log_path = csv_dir / "summary_step_log.csv"

    summary_rows: List[Dict[str, str]] = []
    step_rows: List[Dict[str, str]] = []

    summary_fields = [
        "seed_name",
        "seed_value",
        "fraction",
        "train_samples",
        "train_total",
        "maml_4cls_macro_f1",
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

    print("========== Auto Seed×Fraction MAML 4cls ==========")
    print(f"python_exe : {python_exe}")
    print(f"src_dir    : {src_dir}")
    print(f"out_root   : {out_root}")
    print(f"csv_dir    : {csv_dir}")
    print(f"stdout_dir : {stdout_dir}")
    print(f"tmp_ckpt   : {tmp_ckpt_root}")
    print(f"seed_list  : {seeds}")
    print(f"fractions  : {fracs}")
    print(f"dry_run    : {dry_run}")
    print("===================================================")

    for seed_name, seed_value in seeds:
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
            status, err = "ok", ""
            try:
                run_cmd(cmd, cwd=src_dir, stdout_file=stdout_dir / f"{seed_tag}__{step_name}.log", dry_run=dry_run)
            except Exception as e:
                status, err = "fail", str(e)
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
            data_root = Path(cfg.DATASET_STFT)
            snapshot = MAMLTrainSnapshot(data_root=data_root) if not dry_run else None
            n_total = snapshot.n_total if snapshot is not None else 0
            perm = np.random.default_rng(seed_value).permutation(n_total) if n_total > 0 else np.array([], dtype=np.int64)

            for frac in fracs:
                row = {
                    "seed_name": seed_name,
                    "seed_value": str(seed_value),
                    "fraction": f"{frac:.1f}",
                    "train_samples": "",
                    "train_total": str(n_total),
                    "maml_4cls_macro_f1": "",
                    "status": "ok",
                    "error": "",
                }

                n_keep = max(1, int(round(n_total * float(frac)))) if n_total > 0 else 0
                idx = perm[:n_keep] if n_total > 0 else np.array([], dtype=np.int64)
                if snapshot is not None:
                    snapshot.apply_fraction(idx)

                row["train_samples"] = str(n_keep)
                print(f"\n-- {seed_tag} | fraction={frac:.1f} | train={n_keep}/{n_total} --")

                frac_ckpt_dir = tmp_ckpt_root / seed_tag / frac_tag(frac)
                frac_ckpt_dir.mkdir(parents=True, exist_ok=True)

                t0 = time.time()
                ts0 = dt.datetime.now().isoformat(timespec="seconds")
                status, err = "ok", ""
                try:
                    cmd = [
                        python_exe,
                        "run_maml_4cls.py",
                        "--data_root",
                        str(cfg.DATASET_STFT),
                        "--save_dir",
                        str(frac_ckpt_dir),
                        "--log_dir",
                        str(csv_dir),
                        "--seed",
                        str(seed_value),
                        "--epochs",
                        str(int(args.maml_epochs)),
                        "--train_episodes",
                        str(int(args.maml_train_episodes)),
                        "--val_episodes",
                        str(int(args.maml_val_episodes)),
                        "--test_episodes",
                        str(int(args.maml_test_episodes)),
                        "--meta_batch",
                        str(int(args.maml_meta_batch)),
                        "--num_workers",
                        str(int(args.maml_num_workers)),
                    ]

                    before_csv = list(csv_dir.glob("metrics_maml_darknet19_4cls_*.csv"))
                    run_cmd(
                        cmd,
                        cwd=src_dir,
                        stdout_file=stdout_dir / f"{seed_tag}__{frac_tag(frac)}__maml_4cls.log",
                        dry_run=dry_run,
                    )

                    if not dry_run:
                        new_csv = pick_new_file(csv_dir, "metrics_maml_darknet19_4cls_*.csv", before_csv)
                        tagged = tag_csv_name(new_csv, seed_name, seed_value, frac, "maml_4cls")
                        metric = parse_metric_from_csv(tagged, ["test_macro_f1"])
                        row["maml_4cls_macro_f1"] = f"{metric:.6f}"
                except Exception as e:
                    status, err = "fail", str(e)
                    row["status"], row["error"] = "fail", err

                t1 = time.time()
                ts1 = dt.datetime.now().isoformat(timespec="seconds")
                step_rows.append(
                    {
                        "seed_name": seed_name,
                        "seed_value": str(seed_value),
                        "fraction": f"{frac:.1f}",
                        "step": "maml_4cls",
                        "status": status,
                        "duration_sec": f"{(t1 - t0):.2f}",
                        "error": err,
                        "start_time": ts0,
                        "end_time": ts1,
                    }
                )
                dump_csv(step_log_path, step_fields, step_rows)

                summary_rows.append(row)
                dump_csv(summary_path, summary_fields, summary_rows)

                if not dry_run:
                    shutil.rmtree(frac_ckpt_dir, ignore_errors=True)
                    dump_csv(
                        mean_sd_path,
                        ["fraction", "n", "macro_f1_mean", "macro_f1_sd", "macro_f1_mean_pm_sd"],
                        aggregate_mean_sd_by_fraction(summary_rows),
                    )

                if row["status"] == "ok":
                    print(f"[DONE] {seed_tag} fr={frac:.1f} | macro-f1={row['maml_4cls_macro_f1']}")
                else:
                    print(f"[WARN] {seed_tag} fr={frac:.1f} failed | {row['error']}")

        finally:
            if snapshot is not None:
                snapshot.restore()

    if not dry_run:
        dump_csv(
            mean_sd_path,
            ["fraction", "n", "macro_f1_mean", "macro_f1_sd", "macro_f1_mean_pm_sd"],
            aggregate_mean_sd_by_fraction(summary_rows),
        )

    print("\nAll runs finished.")
    print(f"Summary per run : {summary_path}")
    print(f"Summary mean±SD : {mean_sd_path}")
    print(f"Step log        : {step_log_path}")


if __name__ == "__main__":
    main()
