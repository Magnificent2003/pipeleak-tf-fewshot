import argparse
import csv
import os
import random
import time
from typing import Dict

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

import config as cfg


EPS = 1e-8


class NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)


class HierConsMFCCMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, p_drop: float = 0.2, num_classes: int = 4):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
        )
        self.head_parent = nn.Linear(hidden, 2)
        self.head_child = nn.Linear(hidden, num_classes)

    def forward(self, x):
        z = self.backbone(x)
        logits_parent = self.head_parent(z)
        logits_child = self.head_child(z)
        return logits_parent, logits_child


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mfcc_stats_1d(
    sig: np.ndarray,
    sr: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    window: str,
    n_mels: int,
    n_mfcc: int,
    fmin: float,
    fmax: float,
    center: bool,
):
    x = np.asarray(sig, dtype=np.float32).reshape(-1)
    mfcc = librosa.feature.mfcc(
        y=x,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        center=center,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0).astype(np.float32)


def extract_mfcc_stats_split(X: np.ndarray, args, tag: str):
    feats = []
    for i in range(X.shape[0]):
        feats.append(
            mfcc_stats_1d(
                X[i],
                sr=args.sr,
                n_fft=args.n_fft,
                win_length=args.win_length,
                hop_length=args.hop_length,
                window=args.window,
                n_mels=args.n_mels,
                n_mfcc=args.n_mfcc,
                fmin=args.fmin,
                fmax=args.fmax,
                center=bool(args.center),
            )
        )
        if (i + 1) % 200 == 0 or (i + 1) == X.shape[0]:
            print(f"[MFCC-{tag}] {i+1}/{X.shape[0]} done")
    return np.vstack(feats).astype(np.float32)


def parent_metrics_from_cm4(cm4: np.ndarray):
    cm_parent = np.array(
        [
            [cm4[0:2, 0:2].sum(), cm4[0:2, 2:4].sum()],
            [cm4[2:4, 0:2].sum(), cm4[2:4, 2:4].sum()],
        ],
        dtype=np.int64,
    )
    tp = int(cm_parent[1, 1])
    fp = int(cm_parent[0, 1])
    fn = int(cm_parent[1, 0])
    parent_rec = tp / max(tp + fn, 1)
    parent_f1 = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    return cm_parent, float(parent_f1), float(parent_rec)


def mc_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    _, parent_f1, parent_rec = parent_metrics_from_cm4(cm)
    return {
        "acc": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1),
        "parent_f1": float(parent_f1),
        "parent_recall": float(parent_rec),
    }


def consistency_loss(
    logits_parent: torch.Tensor,
    logits_child: torch.Tensor,
    child_to_parent_mat: torch.Tensor,
    loss_type: str,
    tau: float,
):
    lp = logits_parent / tau
    lc = logits_child / tau
    p_parent = F.softmax(lp, dim=1)
    p_child = F.softmax(lc, dim=1)
    p_child_to_parent = torch.matmul(p_child, child_to_parent_mat.t())

    if loss_type == "sym_kl":
        kl1 = F.kl_div(torch.log(p_parent + EPS), p_child_to_parent, reduction="batchmean")
        kl2 = F.kl_div(torch.log(p_child_to_parent + EPS), p_parent, reduction="batchmean")
        return 0.5 * (kl1 + kl2)
    if loss_type == "js":
        m = 0.5 * (p_parent + p_child_to_parent)
        js1 = F.kl_div(torch.log(p_parent + EPS), m, reduction="batchmean")
        js2 = F.kl_div(torch.log(p_child_to_parent + EPS), m, reduction="batchmean")
        return 0.5 * (js1 + js2)
    if loss_type == "l2":
        return F.mse_loss(p_parent, p_child_to_parent, reduction="mean")
    raise ValueError(f"Unsupported consistency loss type: {loss_type}")


def hierarchical_decode(logits_parent: torch.Tensor, logits_child: torch.Tensor, parent_map: torch.Tensor):
    parent_pred = logits_parent.argmax(dim=1)
    allow = (parent_map.view(1, -1) == parent_pred.view(-1, 1))
    masked = logits_child + torch.where(
        allow,
        torch.zeros_like(logits_child),
        torch.full_like(logits_child, -1e9),
    )
    return masked.argmax(dim=1)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion_parent,
    criterion_child,
    parent_map,
    child_to_parent_mat,
    lambda_child: float,
    lambda_cons: float,
    cons_loss: str,
    cons_tau: float,
    use_consistency: bool,
    device: str,
):
    model.train()
    loss_sum = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_parent = parent_map[y]
        optimizer.zero_grad(set_to_none=True)

        logits_parent, logits_child = model(x)
        loss_sup = criterion_parent(logits_parent, y_parent) + lambda_child * criterion_child(logits_child, y)
        if use_consistency:
            loss_cons = consistency_loss(logits_parent, logits_child, child_to_parent_mat, cons_loss, cons_tau)
            loss = loss_sup + lambda_cons * loss_cons
        else:
            loss = loss_sup

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.item()) * y.size(0)
        n += y.size(0)
    return loss_sum / max(n, 1)


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion_parent,
    criterion_child,
    parent_map,
    child_to_parent_mat,
    lambda_child: float,
    lambda_cons: float,
    cons_loss: str,
    cons_tau: float,
    use_consistency: bool,
    num_classes: int,
    device: str,
):
    model.eval()
    loss_sum = 0.0
    n = 0
    y_true_all, y_pred_all = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_parent = parent_map[y]

        logits_parent, logits_child = model(x)
        loss_sup = criterion_parent(logits_parent, y_parent) + lambda_child * criterion_child(logits_child, y)
        if use_consistency:
            loss_cons = consistency_loss(logits_parent, logits_child, child_to_parent_mat, cons_loss, cons_tau)
            loss = loss_sup + lambda_cons * loss_cons
        else:
            loss = loss_sup

        y_pred = hierarchical_decode(logits_parent, logits_child, parent_map)
        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(y_pred.detach().cpu().numpy())

        loss_sum += float(loss.item()) * y.size(0)
        n += y.size(0)

    y_true = np.concatenate(y_true_all) if y_true_all else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_all) if y_pred_all else np.array([], dtype=np.int64)
    mets = mc_metrics(y_true, y_pred, num_classes)
    return loss_sum / max(n, 1), mets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET)
    ap.add_argument("--label_prefix", type=str, default="y4")
    ap.add_argument("--sr", type=int, default=getattr(cfg, "FS", 8192))
    ap.add_argument("--n_fft", type=int, default=256)
    ap.add_argument("--win_length", type=int, default=256)
    ap.add_argument("--hop_length", type=int, default=128)
    ap.add_argument("--window", type=str, default="hamming")
    ap.add_argument("--n_mels", type=int, default=40)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=-1.0)
    ap.add_argument("--center", type=int, default=1)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--batch_size_eval", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=200)

    ap.add_argument("--lambda_child", type=float, default=0.3)
    ap.add_argument("--lambda_cons", type=float, default=0.1)
    ap.add_argument("--cons_loss", type=str, default="sym_kl", choices=["sym_kl", "js", "l2"])
    ap.add_argument("--cons_tau", type=float, default=1.0)
    ap.add_argument("--cons_start_epoch", type=int, default=100)

    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--save_ckpt", type=int, default=0, choices=[0, 1])
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fmax <= 0:
        args.fmax = float(args.sr) / 2.0
    if args.fmin < 0 or args.fmin >= args.fmax:
        raise ValueError(f"Invalid fmin/fmax: fmin={args.fmin}, fmax={args.fmax}")
    if args.cons_tau <= 0:
        raise ValueError("--cons_tau must be > 0.")

    os.makedirs(args.log_dir, exist_ok=True)
    if bool(args.save_ckpt):
        os.makedirs(args.save_dir, exist_ok=True)

    Xtr = np.load(os.path.join(args.data_root, "X_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "X_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "X_test.npy"))
    ytr = np.load(os.path.join(args.data_root, f"{args.label_prefix}_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, f"{args.label_prefix}_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, f"{args.label_prefix}_test.npy")).astype(np.int64)

    num_classes = int(max(np.max(ytr), np.max(yva), np.max(yte))) + 1
    if num_classes != 4:
        raise ValueError(f"This script is for 4-class hierarchy, got num_classes={num_classes}")

    print("Extracting MFCC mean+std features ...")
    Ftr = extract_mfcc_stats_split(Xtr, args, tag="train")
    Fva = extract_mfcc_stats_split(Xva, args, tag="val")
    Fte = extract_mfcc_stats_split(Xte, args, tag="test")

    mu = Ftr.mean(axis=0, keepdims=True)
    sg = Ftr.std(axis=0, keepdims=True) + 1e-6
    Ftr = (Ftr - mu) / sg
    Fva = (Fva - mu) / sg
    Fte = (Fte - mu) / sg

    tr_loader = DataLoader(NpyDataset(Ftr, ytr), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(NpyDataset(Fva, yva), batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(NpyDataset(Fte, yte), batch_size=args.batch_size_eval, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    parent_map = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)
    child_to_parent_mat = torch.zeros(2, num_classes, dtype=torch.float32, device=device)
    for k in range(num_classes):
        child_to_parent_mat[parent_map[k], k] = 1.0

    model = HierConsMFCCMLP(Ftr.shape[1], hidden=args.hidden, p_drop=args.dropout, num_classes=num_classes).to(device)
    counts_c = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    counts_c[counts_c == 0] = counts_c[counts_c > 0].min() if (counts_c > 0).any() else 1.0
    child_w = (counts_c.max() / counts_c) ** 0.5
    criterion_child = nn.CrossEntropyLoss(weight=torch.tensor(child_w, dtype=torch.float32, device=device))

    ytr_parent = np.array([0 if y in [0, 1] else 1 for y in ytr], dtype=np.int64)
    counts_p = np.bincount(ytr_parent, minlength=2).astype(np.float32)
    counts_p[counts_p == 0] = counts_p[counts_p > 0].min() if (counts_p > 0).any() else 1.0
    parent_w = (counts_p.max() / counts_p) ** 0.5
    criterion_parent = nn.CrossEntropyLoss(weight=torch.tensor(parent_w, dtype=torch.float32, device=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_mlp_mfcc_hcl_sched_4cls_{stamp}.csv")
    ckpt_path = os.path.join(args.save_dir, f"mlp_mfcc_hcl_sched_4cls_best_{stamp}.pt")

    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_macro_recall",
        "val_parent_f1",
        "val_parent_recall",
        "cons_enabled",
        "lambda_cons",
        "test_acc",
        "test_macro_f1",
        "test_macro_recall",
        "test_parent_f1",
        "test_parent_recall",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        best_f1 = -1.0
        best_state = None

        for ep in range(1, args.epochs + 1):
            use_consistency = ep >= args.cons_start_epoch
            tr_loss = train_one_epoch(
                model=model,
                loader=tr_loader,
                optimizer=optimizer,
                criterion_parent=criterion_parent,
                criterion_child=criterion_child,
                parent_map=parent_map,
                child_to_parent_mat=child_to_parent_mat,
                lambda_child=args.lambda_child,
                lambda_cons=args.lambda_cons,
                cons_loss=args.cons_loss,
                cons_tau=args.cons_tau,
                use_consistency=use_consistency,
                device=device,
            )
            scheduler.step()

            va_loss, va_m = evaluate(
                model=model,
                loader=va_loader,
                criterion_parent=criterion_parent,
                criterion_child=criterion_child,
                parent_map=parent_map,
                child_to_parent_mat=child_to_parent_mat,
                lambda_child=args.lambda_child,
                lambda_cons=args.lambda_cons,
                cons_loss=args.cons_loss,
                cons_tau=args.cons_tau,
                use_consistency=use_consistency,
                num_classes=num_classes,
                device=device,
            )
            print(
                f"Epoch {ep:03d}/{args.epochs} | train_loss={tr_loss:.4f} | "
                f"val_macro_f1={va_m['f1_macro']:.4f} | val_macro_recall={va_m['recall_macro']:.4f} | "
                f"cons={'ON' if use_consistency else 'OFF'}"
            )

            writer.writerow(
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "val_loss": va_loss,
                    "val_acc": va_m["acc"],
                    "val_macro_f1": va_m["f1_macro"],
                    "val_macro_recall": va_m["recall_macro"],
                    "val_parent_f1": va_m["parent_f1"],
                    "val_parent_recall": va_m["parent_recall"],
                    "cons_enabled": int(use_consistency),
                    "lambda_cons": args.lambda_cons if use_consistency else 0.0,
                    "test_acc": "",
                    "test_macro_f1": "",
                    "test_macro_recall": "",
                    "test_parent_f1": "",
                    "test_parent_recall": "",
                }
            )

            if va_m["f1_macro"] > best_f1:
                best_f1 = va_m["f1_macro"]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state, strict=True)

        te_loss, te_m = evaluate(
            model=model,
            loader=te_loader,
            criterion_parent=criterion_parent,
            criterion_child=criterion_child,
            parent_map=parent_map,
            child_to_parent_mat=child_to_parent_mat,
            lambda_child=args.lambda_child,
            lambda_cons=args.lambda_cons,
            cons_loss=args.cons_loss,
            cons_tau=args.cons_tau,
            use_consistency=(args.epochs >= args.cons_start_epoch),
            num_classes=num_classes,
            device=device,
        )
        print(
            f"[TEST] loss={te_loss:.4f} | acc={te_m['acc']:.4f} | "
            f"macro-F1={te_m['f1_macro']:.4f} | macro-Recall={te_m['recall_macro']:.4f}"
        )

        writer.writerow(
            {
                "epoch": "best",
                "train_loss": "",
                "val_loss": "",
                "val_acc": "",
                "val_macro_f1": best_f1,
                "val_macro_recall": "",
                "val_parent_f1": "",
                "val_parent_recall": "",
                "cons_enabled": int(args.epochs >= args.cons_start_epoch),
                "lambda_cons": args.lambda_cons if args.epochs >= args.cons_start_epoch else 0.0,
                "test_acc": te_m["acc"],
                "test_macro_f1": te_m["f1_macro"],
                "test_macro_recall": te_m["recall_macro"],
                "test_parent_f1": te_m["parent_f1"],
                "test_parent_recall": te_m["parent_recall"],
            }
        )

    if bool(args.save_ckpt):
        torch.save(
            {
                "model": "HierConsMFCCMLP_sched_4cls",
                "state_dict": best_state,
                "cfg": vars(args),
                "feat_mu": mu.squeeze(0).astype(np.float32),
                "feat_sigma": sg.squeeze(0).astype(np.float32),
            },
            ckpt_path,
        )
        print(f"[CKPT] {ckpt_path}")
    print(f"[LOG]  {log_path}")


if __name__ == "__main__":
    main()
