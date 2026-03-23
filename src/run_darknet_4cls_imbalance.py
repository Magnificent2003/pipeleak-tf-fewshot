import argparse
import csv
import os
import random
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset

import config as cfg
from Darknet19 import Darknet19
from NpyDataset import NpyDataset


class FocalLossMultiClass(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        if alpha.ndim != 1:
            raise ValueError(f"alpha should be 1-D, got shape={tuple(alpha.shape)}")
        self.register_buffer("alpha", alpha.float())
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = torch.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        idx = target.view(-1, 1)
        logpt = logp.gather(1, idx).squeeze(1)
        pt = p.gather(1, idx).squeeze(1)
        at = self.alpha[target]
        loss = -at * ((1.0 - pt) ** self.gamma) * logpt

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def confusion_matrix_np(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        ti, pi = int(t), int(p)
        if 0 <= ti < num_classes and 0 <= pi < num_classes:
            cm[ti, pi] += 1
    return cm


def valve_parent_metrics_from_cm4(cm4: np.ndarray) -> Dict[str, float]:
    # Valve class = y4==1
    tp = int(cm4[1, 1])
    fn = int(cm4[1, :].sum() - tp)
    fp = int(cm4[:, 1].sum() - tp)
    valve_recall = tp / max(tp + fn, 1)
    valve_f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)

    # Cross-parent error: true {2,3} predicted {0,1}
    num = int(cm4[2, 0] + cm4[2, 1] + cm4[3, 0] + cm4[3, 1])
    den = int(cm4[2, :].sum() + cm4[3, :].sum())
    cross_parent_err = num / max(den, 1)
    return {
        "valve_recall": float(valve_recall),
        "valve_f1": float(valve_f1),
        "cross_parent_err": float(cross_parent_err),
    }


def evaluate_4cls(model, loader, criterion, device, num_classes: int) -> Tuple[float, Dict[str, float]]:
    model.eval()
    loss_sum, n = 0.0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += float(loss.item()) * x.size(0)
            n += x.size(0)

            pred = logits.argmax(dim=1).detach().cpu().numpy()
            y_pred.append(pred)
            y_true.append(y.detach().cpu().numpy())

    if n == 0:
        raise RuntimeError("Empty loader during evaluate.")
    y_true_np = np.concatenate(y_true)
    y_pred_np = np.concatenate(y_pred)
    cm4 = confusion_matrix_np(y_true_np, y_pred_np, num_classes=num_classes)
    vp = valve_parent_metrics_from_cm4(cm4)

    mets = {
        "acc": float(accuracy_score(y_true_np, y_pred_np)),
        "macro_f1": float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "valve_recall": vp["valve_recall"],
        "valve_f1": vp["valve_f1"],
        "cross_parent_err": vp["cross_parent_err"],
    }
    return loss_sum / max(n, 1), mets


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
    return loss_sum / max(n, 1)


def parse_counts(text: str, num_classes: int) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) != num_classes:
        raise ValueError(f"train_counts should have {num_classes} ints, got {text}")
    arr = np.array([int(p) for p in parts], dtype=np.int64)
    if np.any(arr <= 0):
        raise ValueError(f"train_counts must be positive, got {arr.tolist()}")
    return arr


def select_fixed_counts_per_class(y: np.ndarray, counts: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    selected = []
    for cls, need in enumerate(counts.tolist()):
        idx = np.where(y == cls)[0]
        if len(idx) < need:
            raise ValueError(f"class {cls} has only {len(idx)} samples, but need {need}")
        pick = rng.choice(idx, size=need, replace=False)
        selected.append(pick)
    out = np.concatenate(selected, axis=0)
    rng.shuffle(out)
    return out.astype(np.int64)


def build_class_weights_from_counts(counts: np.ndarray) -> np.ndarray:
    # inverse frequency then normalized to average weight = 1
    inv = 1.0 / np.maximum(counts.astype(np.float64), 1.0)
    w = inv * (len(inv) / np.sum(inv))
    return w.astype(np.float32)


def main():
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
    ap.add_argument("--early_stop", type=int, default=0)
    ap.add_argument("--seed", type=int, default=int(getattr(cfg, "SEED", 2023)))
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--save_ckpt", type=int, default=1)

    ap.add_argument("--loss_type", type=str, default="ce", choices=["ce", "weighted_ce", "focal"])
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--train_counts", type=str, default="157,32,79,110")
    args = ap.parse_args()

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    if bool(args.save_ckpt):
        os.makedirs(args.save_dir, exist_ok=True)

    num_classes = int(args.num_classes)
    if num_classes != 4:
        raise ValueError("This script is designed for 4-class setting (num_classes=4).")

    xtr = os.path.join(args.data_root, f"X_train_stft_{args.img_size}.npy")
    xva = os.path.join(args.data_root, f"X_val_stft_{args.img_size}.npy")
    xte = os.path.join(args.data_root, f"X_test_stft_{args.img_size}.npy")
    ytr = os.path.join(args.data_root, f"{args.label_prefix}_train.npy")
    yva = os.path.join(args.data_root, f"{args.label_prefix}_val.npy")
    yte = os.path.join(args.data_root, f"{args.label_prefix}_test.npy")

    ds_tr_full = NpyDataset(xtr, ytr, normalize="imagenet", memmap=True)
    ds_va = NpyDataset(xva, yva, normalize="imagenet", memmap=True)
    ds_te = NpyDataset(xte, yte, normalize="imagenet", memmap=True)
    y_tr_full = ds_tr_full.y.astype(np.int64)

    desired_counts = parse_counts(args.train_counts, num_classes=num_classes)
    selected_idx = select_fixed_counts_per_class(y_tr_full, desired_counts, seed=int(args.seed))
    ds_tr = Subset(ds_tr_full, selected_idx.tolist())

    y_sel = y_tr_full[selected_idx]
    sel_counts = np.bincount(y_sel, minlength=num_classes).astype(np.int64)
    class_w_np = build_class_weights_from_counts(sel_counts)
    class_w_t = torch.tensor(class_w_np, dtype=torch.float32, device=device)

    print(f"[Seed] {args.seed}")
    print(f"[Train subset] selected={len(selected_idx)} / full={len(ds_tr_full)}")
    print(f"[Train subset y4 counts] {sel_counts.tolist()} (target={desired_counts.tolist()})")
    print(f"[Class weights] {class_w_np.tolist()} | mean={float(class_w_np.mean()):.4f}")
    print(f"[Val/Test size] {len(ds_va)}/{len(ds_te)}")

    tr_loader = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    va_loader = DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    te_loader = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    model = Darknet19(num_classes=num_classes).to(device)
    if args.loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_w_t)
    elif args.loss_type == "focal":
        criterion = FocalLossMultiClass(alpha=class_w_t, gamma=float(args.focal_gamma), reduction="mean")
    else:
        raise ValueError(f"Unsupported loss_type={args.loss_type}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_name = f"metrics_darknet19_4cls_imbalance_{args.loss_type}_{stamp}.csv"
    log_path = os.path.join(args.log_dir, log_name)
    ckpt_path = os.path.join(args.save_dir, f"darknet19_4cls_imbalance_{args.loss_type}_best_{stamp}.pth")

    fields = [
        "epoch",
        "loss_type",
        "train_n",
        "train_counts",
        "class_weights",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_valve_recall",
        "val_valve_f1",
        "val_cross_parent_err",
        "test_acc",
        "test_macro_f1",
        "test_valve_recall",
        "test_valve_f1",
        "test_cross_parent_err",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

    best_val_valve_f1 = -1.0
    best_state = None
    patience, noimp = int(args.early_stop), 0

    for ep in range(1, int(args.epochs) + 1):
        tr_loss = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_m = evaluate_4cls(model, va_loader, criterion, device, num_classes=num_classes)
        scheduler.step()

        print(
            f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} "
            f"| val_macro_f1={va_m['macro_f1']:.4f} | valve_R={va_m['valve_recall']:.4f} "
            f"| valve_F1={va_m['valve_f1']:.4f} | cross_parent_err={va_m['cross_parent_err']:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writerow(
                {
                    "epoch": ep,
                    "loss_type": args.loss_type,
                    "train_n": len(selected_idx),
                    "train_counts": ",".join(map(str, sel_counts.tolist())),
                    "class_weights": ",".join([f"{x:.6f}" for x in class_w_np.tolist()]),
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "val_acc": va_m["acc"],
                    "val_macro_f1": va_m["macro_f1"],
                    "val_valve_recall": va_m["valve_recall"],
                    "val_valve_f1": va_m["valve_f1"],
                    "val_cross_parent_err": va_m["cross_parent_err"],
                    "test_acc": "",
                    "test_macro_f1": "",
                    "test_valve_recall": "",
                    "test_valve_f1": "",
                    "test_cross_parent_err": "",
                }
            )

        if va_m["valve_f1"] > best_val_valve_f1:
            best_val_valve_f1 = va_m["valve_f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            noimp = 0
            if bool(args.save_ckpt):
                torch.save(
                    {
                        "state_dict": best_state,
                        "num_classes": num_classes,
                        "seed": int(args.seed),
                        "loss_type": args.loss_type,
                        "train_counts": sel_counts.tolist(),
                        "class_weights": class_w_np.tolist(),
                    },
                    ckpt_path,
                )
        else:
            noimp += 1
            if patience > 0 and noimp >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    te_loss, te_m = evaluate_4cls(model, te_loader, criterion, device, num_classes=num_classes)
    print(
        f"[TEST] macro_f1={te_m['macro_f1']:.4f} | valve_R={te_m['valve_recall']:.4f} "
        f"| valve_F1={te_m['valve_f1']:.4f} | cross_parent_err={te_m['cross_parent_err']:.4f}"
    )

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writerow(
            {
                "epoch": "best",
                "loss_type": args.loss_type,
                "train_n": len(selected_idx),
                "train_counts": ",".join(map(str, sel_counts.tolist())),
                "class_weights": ",".join([f"{x:.6f}" for x in class_w_np.tolist()]),
                "train_loss": "",
                "val_loss": "",
                "val_acc": "",
                "val_macro_f1": "",
                "val_valve_recall": "",
                "val_valve_f1": best_val_valve_f1,
                "val_cross_parent_err": "",
                "test_acc": te_m["acc"],
                "test_macro_f1": te_m["macro_f1"],
                "test_valve_recall": te_m["valve_recall"],
                "test_valve_f1": te_m["valve_f1"],
                "test_cross_parent_err": te_m["cross_parent_err"],
            }
        )

    print(f"[LOG] {log_path}")
    if bool(args.save_ckpt):
        print(f"[CKPT] {ckpt_path}")


if __name__ == "__main__":
    main()
