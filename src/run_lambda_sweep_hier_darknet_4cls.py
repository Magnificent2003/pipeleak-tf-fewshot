import argparse
import datetime as dt
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

import config as cfg
from HierarchicalDarknet19 import HierarchicalDarknet19
from NpyDataset import NpyDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(float(p))
    if not out:
        raise ValueError("empty float list")
    return out


def parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    if not out:
        raise ValueError("empty int list")
    return out


def get_default_seed_values(runs_per_lambda: int) -> List[int]:
    # Repeated runs on ONE fixed dataset split:
    # these seeds control training randomness only.
    base = int(getattr(cfg, "SEED", 42))
    return [base + i for i in range(runs_per_lambda)]


def build_run_seeds(seed_values_text: str, runs_per_lambda: int) -> List[int]:
    if seed_values_text.strip():
        seeds = parse_int_list(seed_values_text)
    else:
        seeds = get_default_seed_values(runs_per_lambda)

    if len(seeds) < runs_per_lambda:
        raise ValueError(
            f"seed values are fewer than runs_per_lambda: {len(seeds)} < {runs_per_lambda}"
        )
    if len(seeds) > runs_per_lambda:
        seeds = seeds[:runs_per_lambda]
    return seeds


def ensure_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    y_np = y.detach().cpu().numpy()
    p_np = pred.detach().cpu().numpy()

    acc = float((pred == y).float().mean().item())
    macro_f1 = float(f1_score(y_np, p_np, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_np, p_np, average="macro", zero_division=0))

    # Parent mapping for evaluation: classes 0/1 -> 0, classes 2/3 -> 1
    y_parent = (y_np >= 2).astype(np.int64)
    p_parent = (p_np >= 2).astype(np.int64)
    parent_f1 = float(f1_score(y_parent, p_parent, average="binary", zero_division=0))
    parent_recall = float(recall_score(y_parent, p_parent, average="binary", zero_division=0))

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "parent_f1": parent_f1,
        "parent_recall": parent_recall,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion_b: nn.Module,
    criterion_c: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    parent_map: torch.Tensor,
    lambda_child: float,
    epoch: int,
    warmup_child_epochs: int,
) -> float:
    model.train()
    loss_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long)
        y_parent = parent_map[y]

        optimizer.zero_grad(set_to_none=True)
        logits_b, logits_c = model(x)

        # Keep the same strategy as run_hier_darknet_4cls_new.py
        if epoch <= warmup_child_epochs:
            loss = criterion_c(logits_c, y)
        else:
            loss = criterion_b(logits_b, y_parent) + lambda_child * criterion_c(logits_c, y)

        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
    return loss_sum / max(n, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion_b: nn.Module,
    criterion_c: nn.Module,
    device: torch.device,
    parent_map: torch.Tensor,
    lambda_child: float,
) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum = 0.0
    n = 0
    all_logits_c: List[torch.Tensor] = []
    all_y: List[torch.Tensor] = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long)
        y_parent = parent_map[y]

        logits_b, logits_c = model(x)
        loss = criterion_b(logits_b, y_parent) + lambda_child * criterion_c(logits_c, y)

        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
        all_logits_c.append(logits_c)
        all_y.append(y)

    logits_c = torch.cat(all_logits_c, dim=0)
    ys = torch.cat(all_y, dim=0)
    mets = metrics_from_logits(logits_c, ys)
    return loss_sum / max(n, 1), mets


def clone_state_dict_cpu(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def summarize_metric(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "sd": float("nan")}
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return {"mean": mean, "sd": sd}


def summarize_runs(runs: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    ok_runs = [r for r in runs if r.get("status") == "ok"]
    keys = [
        "test_acc",
        "test_macro_f1",
        "test_macro_recall",
        "test_parent_f1",
        "test_parent_recall",
        "best_val_macro_f1",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = [float(r[k]) for r in ok_runs]
        s = summarize_metric(vals)
        out[k] = {
            "mean": s["mean"],
            "sd": s["sd"],
            "mean_pm_sd": f"{s['mean']:.4f} ± {s['sd']:.4f}",
        }
    return out


def run_single_experiment(
    args: argparse.Namespace,
    seed: int,
    lambda_child: float,
    ds_tr: NpyDataset,
    ds_va: NpyDataset,
    ds_te: NpyDataset,
    device: torch.device,
) -> Dict:
    set_seed(seed)
    t0 = time.time()

    generator = torch.Generator()
    generator.manual_seed(seed)

    tr_loader = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        generator=generator,
    )
    va_loader = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )
    te_loader = DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    model = HierarchicalDarknet19(num_classes=args.num_classes).to(device)
    criterion_b = nn.CrossEntropyLoss()
    criterion_c = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Keep training parent mapping aligned with run_hier_darknet_4cls_new.py
    parent_map = torch.tensor([1, 1, 0, 0], dtype=torch.long, device=device)

    best_val_f1 = -1.0
    best_epoch = 0
    best_state: Dict[str, torch.Tensor] = {}
    noimp = 0
    patience = int(args.early_stop)

    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model=model,
            loader=tr_loader,
            criterion_b=criterion_b,
            criterion_c=criterion_c,
            optimizer=optimizer,
            device=device,
            parent_map=parent_map,
            lambda_child=lambda_child,
            epoch=ep,
            warmup_child_epochs=int(args.warmup_child_epochs),
        )
        va_loss, va_m = evaluate(
            model=model,
            loader=va_loader,
            criterion_b=criterion_b,
            criterion_c=criterion_c,
            device=device,
            parent_map=parent_map,
            lambda_child=lambda_child,
        )
        scheduler.step()

        if ep == 1 or ep % int(args.print_every) == 0 or ep == args.epochs:
            print(
                f"    Epoch {ep:03d}/{args.epochs} | "
                f"train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
                f"val_macro_f1={va_m['macro_f1']:.4f}"
            )

        if va_m["macro_f1"] > best_val_f1:
            best_val_f1 = float(va_m["macro_f1"])
            best_epoch = ep
            best_state = clone_state_dict_cpu(model)
            noimp = 0
        else:
            noimp += 1
            if patience > 0 and noimp >= patience:
                print(f"    [EarlyStop] no improvement for {patience} epochs.")
                break

    if not best_state:
        best_state = clone_state_dict_cpu(model)
        best_epoch = args.epochs
        best_val_f1 = float("nan")

    model.load_state_dict(best_state, strict=True)
    te_loss, te_m = evaluate(
        model=model,
        loader=te_loader,
        criterion_b=criterion_b,
        criterion_c=criterion_c,
        device=device,
        parent_map=parent_map,
        lambda_child=lambda_child,
    )

    duration = time.time() - t0
    out = {
        "status": "ok",
        "error": "",
        "seed": int(seed),
        "lambda_child": float(lambda_child),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_val_f1),
        "test_loss": float(te_loss),
        "test_acc": float(te_m["acc"]),
        "test_macro_f1": float(te_m["macro_f1"]),
        "test_macro_recall": float(te_m["macro_recall"]),
        "test_parent_f1": float(te_m["parent_f1"]),
        "test_parent_recall": float(te_m["parent_recall"]),
        "duration_sec": float(duration),
    }

    # Release some memory between runs.
    del model, optimizer, scheduler, best_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def build_payload(
    args: argparse.Namespace,
    output_json: Path,
    run_seeds: Sequence[int],
    lambda_values: Sequence[float],
    results: Sequence[Dict],
    started_at: str,
    device: torch.device,
    elapsed_sec: float,
) -> Dict:
    return {
        "meta": {
            "script": Path(__file__).name,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "started_at": started_at,
            "project_root": str(Path(__file__).resolve().parents[1]),
            "output_json": str(output_json),
            "device": str(device),
            "data_root": args.data_root,
            "label_prefix": args.label_prefix,
            "num_classes": int(args.num_classes),
            "img_size": int(args.img_size),
            "batch_size": int(args.batch_size),
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "num_workers": int(args.num_workers),
            "early_stop": int(args.early_stop),
            "warmup_child_epochs": int(args.warmup_child_epochs),
            "pin_memory": bool(args.pin_memory),
            "print_every": int(args.print_every),
            "runs_per_lambda": int(args.runs_per_lambda),
            "seed_values": [int(s) for s in run_seeds],
            "lambda_values": [float(x) for x in lambda_values],
            "same_dataset_split": True,
            "split_note": "Uses existing npy files under data_root; does not rebuild/switch dataset split.",
            "elapsed_sec": float(elapsed_sec),
        },
        "results": list(results),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--label_prefix", type=str, default="y4")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", type=int, default=1)
    ap.add_argument("--early_stop", type=int, default=0)
    ap.add_argument("--warmup_child_epochs", type=int, default=150)
    ap.add_argument("--print_every", type=int, default=20)
    ap.add_argument("--runs_per_lambda", type=int, default=5)
    ap.add_argument("--seed_values", type=str, default="")
    ap.add_argument("--lambda_values", type=str, default="0.1,0.3,0.7,1.5,2.5,5.0")
    ap.add_argument("--device", type=str, default="")
    ap.add_argument("--output_json", type=str, default="")
    ap.add_argument("--continue_on_error", type=int, default=0)
    ap.add_argument("--dry_run", type=int, default=0)
    args = ap.parse_args()

    if int(args.num_classes) != 4:
        raise ValueError("this script is designed for 4-class hierarchical Darknet only")
    if int(args.runs_per_lambda) <= 0:
        raise ValueError("runs_per_lambda must be >= 1")
    if int(args.epochs) <= 0:
        raise ValueError("epochs must be >= 1")
    if int(args.print_every) <= 0:
        raise ValueError("print_every must be >= 1")

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent

    lambda_values = parse_float_list(args.lambda_values)
    run_seeds = build_run_seeds(args.seed_values, int(args.runs_per_lambda))

    if args.output_json.strip():
        output_json = Path(args.output_json).resolve()
    else:
        out_dir = project_root / "runs" / f"lambda_sweep_hier_darknet_4cls_{now_str()}"
        output_json = out_dir / "lambda_sweep_results.json"
    output_json.parent.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).resolve()
    xtr = data_root / f"X_train_stft_{args.img_size}.npy"
    xva = data_root / f"X_val_stft_{args.img_size}.npy"
    xte = data_root / f"X_test_stft_{args.img_size}.npy"
    ytr = data_root / f"{args.label_prefix}_train.npy"
    yva = data_root / f"{args.label_prefix}_val.npy"
    yte = data_root / f"{args.label_prefix}_test.npy"
    for p in [xtr, xva, xte, ytr, yva, yte]:
        ensure_file(p)

    if args.device.strip():
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("========== Lambda Sweep (Hier Darknet 4cls) ==========")
    print(f"device         : {device}")
    print(f"data_root      : {data_root}")
    print(f"output_json    : {output_json}")
    print(f"lambda_values  : {lambda_values}")
    print(f"run_seeds      : {run_seeds}")
    print(f"runs_per_lambda: {args.runs_per_lambda}")
    print("split_mode     : fixed existing dataset split (no rebuild)")
    print("======================================================")

    ds_tr = NpyDataset(str(xtr), str(ytr), normalize="imagenet", memmap=True)
    ds_va = NpyDataset(str(xva), str(yva), normalize="imagenet", memmap=True)
    ds_te = NpyDataset(str(xte), str(yte), normalize="imagenet", memmap=True)

    for split_name, ds in [("train", ds_tr), ("val", ds_va), ("test", ds_te)]:
        y_min, y_max = int(np.min(ds.y)), int(np.max(ds.y))
        if y_min < 0 or y_max >= int(args.num_classes):
            print(
                f"[WARN] {split_name} label range {y_min}..{y_max} is out of [0,{args.num_classes - 1}]"
            )
    print(f"Train/Val/Test sizes: {len(ds_tr)}/{len(ds_va)}/{len(ds_te)}")

    if bool(args.dry_run):
        payload = build_payload(
            args=args,
            output_json=output_json,
            run_seeds=run_seeds,
            lambda_values=lambda_values,
            results=[],
            started_at=dt.datetime.now().isoformat(timespec="seconds"),
            device=device,
            elapsed_sec=0.0,
        )
        save_json(output_json, payload)
        print(f"[DRY-RUN] JSON template saved to: {output_json}")
        return

    started_at = dt.datetime.now().isoformat(timespec="seconds")
    global_t0 = time.time()
    all_results: List[Dict] = []
    cont = bool(args.continue_on_error)

    for lambda_child in lambda_values:
        lambda_t0 = time.time()
        print(f"\n---- lambda_child={lambda_child:.4f} ----")
        run_rows: List[Dict] = []

        for run_idx, seed in enumerate(run_seeds, start=1):
            print(f"  [Run {run_idx:02d}/{len(run_seeds)}] seed={seed}")
            try:
                row = run_single_experiment(
                    args=args,
                    seed=seed,
                    lambda_child=float(lambda_child),
                    ds_tr=ds_tr,
                    ds_va=ds_va,
                    ds_te=ds_te,
                    device=device,
                )
                print(
                    "    [TEST] "
                    f"macro_f1={row['test_macro_f1']:.4f} "
                    f"macro_recall={row['test_macro_recall']:.4f} "
                    f"parent_f1={row['test_parent_f1']:.4f} "
                    f"parent_recall={row['test_parent_recall']:.4f}"
                )
                row["run_index"] = run_idx
                run_rows.append(row)
            except Exception as e:
                msg = str(e)
                print(f"    [ERROR] {msg}")
                run_rows.append(
                    {
                        "status": "fail",
                        "error": msg,
                        "seed": int(seed),
                        "lambda_child": float(lambda_child),
                        "run_index": run_idx,
                    }
                )
                if not cont:
                    raise

            partial_payload = build_payload(
                args=args,
                output_json=output_json,
                run_seeds=run_seeds,
                lambda_values=lambda_values,
                results=all_results
                + [
                    {
                        "lambda_child": float(lambda_child),
                        "runs": run_rows,
                        "summary": summarize_runs(run_rows),
                        "elapsed_sec": float(time.time() - lambda_t0),
                    }
                ],
                started_at=started_at,
                device=device,
                elapsed_sec=float(time.time() - global_t0),
            )
            save_json(output_json, partial_payload)

        lambda_result = {
            "lambda_child": float(lambda_child),
            "runs": run_rows,
            "summary": summarize_runs(run_rows),
            "elapsed_sec": float(time.time() - lambda_t0),
        }
        all_results.append(lambda_result)
        print(
            f"  [Summary] lambda_child={lambda_child:.4f} | "
            f"macro_f1={lambda_result['summary']['test_macro_f1']['mean_pm_sd']} | "
            f"macro_recall={lambda_result['summary']['test_macro_recall']['mean_pm_sd']}"
        )

    final_payload = build_payload(
        args=args,
        output_json=output_json,
        run_seeds=run_seeds,
        lambda_values=lambda_values,
        results=all_results,
        started_at=started_at,
        device=device,
        elapsed_sec=float(time.time() - global_t0),
    )
    save_json(output_json, final_payload)
    print(f"\n[DONE] JSON saved to: {output_json}")


if __name__ == "__main__":
    main()
