import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score
from torch.utils.data import DataLoader

import config as cfg


SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MAML import EpisodicSplitDataset, MAMLClassifier, collate_episodes  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_macro_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"acc": 0.0, "macro_f1": 0.0, "macro_recall": 0.0}
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    return {"acc": acc, "macro_f1": macro_f1, "macro_recall": macro_recall}


def run_epoch(
    model: MAMLClassifier,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    inner_args: Dict,
    n_way: int,
    meta_train: bool,
) -> Dict[str, float]:
    if meta_train:
        model.train()
    else:
        model.eval()

    loss_list: List[float] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []

    for x_shot, x_query, y_shot, y_query in loader:
        x_shot = x_shot.to(device, non_blocking=True)
        x_query = x_query.to(device, non_blocking=True)
        y_shot = y_shot.to(device, non_blocking=True)
        y_query = y_query.to(device, non_blocking=True)

        logits = model(x_shot, x_query, y_shot, inner_args=inner_args, meta_train=meta_train)
        logits_flat = logits.reshape(-1, n_way)
        labels_flat = y_query.reshape(-1)

        loss = F.cross_entropy(logits_flat, labels_flat)

        if meta_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        pred_flat = logits_flat.argmax(dim=1)
        loss_list.append(float(loss.item()))
        all_true.append(labels_flat.detach().cpu().numpy())
        all_pred.append(pred_flat.detach().cpu().numpy())

    y_true = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)
    m = safe_macro_metrics(y_true, y_pred)
    m["loss"] = float(np.mean(loss_list)) if loss_list else 0.0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.MAML_DATA_ROOT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--normalize", type=str, default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--n_way", type=int, default=cfg.MAML_N_WAY)
    ap.add_argument("--n_support", type=int, default=cfg.MAML_N_SUPPORT)
    ap.add_argument("--n_query", type=int, default=cfg.MAML_N_QUERY)
    ap.add_argument("--epochs", type=int, default=cfg.MAML_EPOCHS)
    ap.add_argument("--train_episodes", type=int, default=cfg.MAML_TRAIN_EPISODES)
    ap.add_argument("--val_episodes", type=int, default=cfg.MAML_VAL_EPISODES)
    ap.add_argument("--test_episodes", type=int, default=cfg.MAML_TEST_EPISODES)
    ap.add_argument("--meta_batch", type=int, default=cfg.MAML_META_BATCH)
    ap.add_argument("--lr", type=float, default=cfg.MAML_LR)
    ap.add_argument("--weight_decay", type=float, default=cfg.MAML_WEIGHT_DECAY)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--sample_with_replacement", type=int, default=1, choices=[0, 1])
    ap.add_argument("--early_stop", type=int, default=0, help="stop if no val_f1 improvement for N epochs; 0=off")
    # inner-loop args
    ap.add_argument("--inner_steps", type=int, default=cfg.MAML_INNER_STEPS)
    ap.add_argument("--inner_encoder_lr", type=float, default=cfg.MAML_INNER_ENCODER_LR)
    ap.add_argument("--inner_classifier_lr", type=float, default=cfg.MAML_INNER_CLASSIFIER_LR)
    ap.add_argument("--inner_momentum", type=float, default=cfg.MAML_INNER_MOMENTUM)
    ap.add_argument("--inner_weight_decay", type=float, default=cfg.MAML_INNER_WEIGHT_DECAY)
    ap.add_argument("--first_order", type=int, default=cfg.MAML_FIRST_ORDER, choices=[0, 1])
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_train = EpisodicSplitDataset(
        data_root=args.data_root,
        split="train",
        n_way=args.n_way,
        n_shot=args.n_support,
        n_query=args.n_query,
        n_episode=args.train_episodes,
        image_size=args.img_size,
        normalize=args.normalize,
        sample_with_replacement=bool(int(args.sample_with_replacement)),
    )
    ds_val = EpisodicSplitDataset(
        data_root=args.data_root,
        split="val",
        n_way=args.n_way,
        n_shot=args.n_support,
        n_query=args.n_query,
        n_episode=args.val_episodes,
        image_size=args.img_size,
        normalize=args.normalize,
        sample_with_replacement=bool(int(args.sample_with_replacement)),
    )
    ds_test = EpisodicSplitDataset(
        data_root=args.data_root,
        split="test",
        n_way=args.n_way,
        n_shot=args.n_support,
        n_query=args.n_query,
        n_episode=args.test_episodes,
        image_size=args.img_size,
        normalize=args.normalize,
        sample_with_replacement=bool(int(args.sample_with_replacement)),
    )

    tr_loader = DataLoader(
        ds_train,
        batch_size=int(args.meta_batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_episodes,
    )
    va_loader = DataLoader(
        ds_val,
        batch_size=int(args.meta_batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_episodes,
    )
    te_loader = DataLoader(
        ds_test,
        batch_size=int(args.meta_batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        collate_fn=collate_episodes,
    )

    model = MAMLClassifier(n_way=int(args.n_way)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    inner_args = {
        "n_step": int(args.inner_steps),
        "encoder_lr": float(args.inner_encoder_lr),
        "classifier_lr": float(args.inner_classifier_lr),
        "momentum": float(args.inner_momentum),
        "weight_decay": float(args.inner_weight_decay),
        "first_order": bool(int(args.first_order)),
        "frozen": [],
    }

    print(f"Data root: {args.data_root}")
    print(f"Train class counts: {ds_train.class_counts}")
    print(f"Val class counts  : {ds_val.class_counts}")
    print(f"Test class counts : {ds_test.class_counts}")
    print("Protocol: train uses train split only, selection uses val split, final report uses test split.")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(args.save_dir, f"maml_darknet19_4cls_best_{stamp}.pth")
    log_path = os.path.join(args.log_dir, f"metrics_maml_darknet19_4cls_{stamp}.csv")

    fieldnames = [
        "epoch",
        "train_loss",
        "train_acc",
        "train_macro_f1",
        "train_macro_recall",
        "val_loss",
        "val_acc",
        "val_macro_f1",
        "val_macro_recall",
        "test_loss",
        "test_acc",
        "test_macro_f1",
        "test_macro_recall",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    best_val_f1 = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, int(args.epochs) + 1):
        tr = run_epoch(
            model=model,
            loader=tr_loader,
            optimizer=optimizer,
            device=device,
            inner_args=inner_args,
            n_way=int(args.n_way),
            meta_train=True,
        )
        va = run_epoch(
            model=model,
            loader=va_loader,
            optimizer=optimizer,
            device=device,
            inner_args=inner_args,
            n_way=int(args.n_way),
            meta_train=False,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"train_f1={tr['macro_f1']:.4f} train_rec={tr['macro_recall']:.4f} | "
            f"val_f1={va['macro_f1']:.4f} val_rec={va['macro_recall']:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(
                {
                    "epoch": epoch,
                    "train_loss": tr["loss"],
                    "train_acc": tr["acc"],
                    "train_macro_f1": tr["macro_f1"],
                    "train_macro_recall": tr["macro_recall"],
                    "val_loss": va["loss"],
                    "val_acc": va["acc"],
                    "val_macro_f1": va["macro_f1"],
                    "val_macro_recall": va["macro_recall"],
                    "test_loss": "",
                    "test_acc": "",
                    "test_macro_f1": "",
                    "test_macro_recall": "",
                }
            )

        if va["macro_f1"] > best_val_f1:
            best_val_f1 = va["macro_f1"]
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "inner_args": inner_args,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"** Saved best (val_macro_f1={best_val_f1:.4f}) -> {ckpt_path}")
        else:
            stale += 1
            if int(args.early_stop) > 0 and stale >= int(args.early_stop):
                print(f"[EarlyStop] no val_macro_f1 improvement for {args.early_stop} epochs.")
                break

    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["state_dict"], strict=True)
        best_epoch = int(ck.get("best_epoch", best_epoch))
        best_val_f1 = float(ck.get("best_val_f1", best_val_f1))

    te = run_epoch(
        model=model,
        loader=te_loader,
        optimizer=optimizer,
        device=device,
        inner_args=inner_args,
        n_way=int(args.n_way),
        meta_train=False,
    )

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(
            {
                "epoch": "best",
                "train_loss": "",
                "train_acc": "",
                "train_macro_f1": "",
                "train_macro_recall": "",
                "val_loss": "",
                "val_acc": "",
                "val_macro_f1": best_val_f1,
                "val_macro_recall": "",
                "test_loss": te["loss"],
                "test_acc": te["acc"],
                "test_macro_f1": te["macro_f1"],
                "test_macro_recall": te["macro_recall"],
            }
        )

    print(
        f"[TEST] best_ep={best_epoch:03d} "
        f"macro-f1={te['macro_f1']:.4f} macro-recall={te['macro_recall']:.4f} acc={te['acc']:.4f}"
    )
    print(f"[LOG] {log_path}")
    print(f"[CKPT] {ckpt_path}")


if __name__ == "__main__":
    main()
