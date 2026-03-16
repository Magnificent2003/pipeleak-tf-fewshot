import os, csv, argparse, time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import config as cfg

from typing import Dict
from torch.utils.data import DataLoader, Subset
from NpyDataset import NpyDataset
from sklearn.metrics import f1_score
from HierarchicalDarknet19 import HierarchicalDarknet19

def set_seed(seed: int):
    """固定所有能控制到的随机行为，便于实验复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- train and eval ----------
@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    acc = float((pred == y).float().mean().item())

    y_np = y.cpu().numpy()
    p_np = pred.cpu().numpy()
    avg = "binary" if num_classes == 2 else "macro"
    f1 = float(f1_score(y_np, p_np, average=avg, zero_division=0))

    return {"acc": acc, "f1": f1}

def train_one_epoch(model, loader, criterion_b, criterion_c, optimizer, device, parent_map, lambda_child):
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long)
        y_parent = parent_map[y]

        optimizer.zero_grad(set_to_none=True)
        logits_b, logits_c = model(x)
        loss = criterion_b(logits_b, y_parent) + lambda_child * criterion_c(logits_c, y)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
    return loss_sum / max(n, 1)

def evaluate(model, loader, criterion_b, criterion_c, device, num_classes, parent_map, lambda_child):
    model.eval()
    loss_sum, n = 0.0, 0
    all_logits_c, all_y = [], []
    with torch.no_grad():
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
    mets = metrics_from_logits(logits_c, ys, num_classes)
    return loss_sum / max(n, 1), mets

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
    ap.add_argument("--fraction", type=float, default=cfg.TRAIN_FRACTION)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--early_stop", type=int, default=0)
    ap.add_argument("--lambda_child", type=float, default=0.3)
    args = ap.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 路径
    Xtr = os.path.join(args.data_root, f"X_train_stft_{args.img_size}.npy")
    Xva = os.path.join(args.data_root, f"X_val_stft_{args.img_size}.npy")
    Xte = os.path.join(args.data_root, f"X_test_stft_{args.img_size}.npy")
    ytr = os.path.join(args.data_root, f"{args.label_prefix}_train.npy")
    yva = os.path.join(args.data_root, f"{args.label_prefix}_val.npy")
    yte = os.path.join(args.data_root, f"{args.label_prefix}_test.npy")

    # 数据集
    ds_tr = NpyDataset(Xtr, ytr, normalize="imagenet", memmap=True)
    ds_va = NpyDataset(Xva, yva, normalize="imagenet", memmap=True)
    ds_te = NpyDataset(Xte, yte, normalize="imagenet", memmap=True)

    # 先在完整训练集上统计类别数
    num_classes = int(np.max(ds_tr.y)) + 1
    
    # ===== 训练集采样比例（TRAIN_FRACTION） =====
    train_fraction = args.fraction
    train_fraction = max(0.0, min(1.0, train_fraction))

    n_total = len(ds_tr)
    n_keep = max(1, int(round(n_total * train_fraction)))

    seed = int(getattr(cfg, "RANDOM_SEED", 42))
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_total)[:n_keep]

    ds_tr = Subset(ds_tr, indices)
    print(f"[Train subset] Use {n_keep}/{n_total} samples "
        f"({train_fraction:.2f} of training data).")

    print(f"Train/Val/Test sizes: {len(ds_tr)}/{len(ds_va)}/{len(ds_te)} | classes={num_classes}")

    tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    # 模型 & 损失
    model = HierarchicalDarknet19(num_classes=num_classes).to(device)
    criterion_b = nn.CrossEntropyLoss()
    criterion_c = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 父类映射：四分类标签 → 父类（0=non-leak, 1=leak）
    PARENT_MAP = torch.tensor([1, 1, 0, 0], dtype=torch.long).to(device)  # idx 0,1,2,3

    # 日志
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_darknet19_hier_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
        w.writeheader()

    best_val_f1 = -1.0
    best_ckpt = os.path.join(args.save_dir, f"darknet19_hier_best_{stamp}.pth")
    patience, noimp = args.early_stop, 0

    # 训练
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, tr_loader, criterion_b, criterion_c, optimizer, device,
            PARENT_MAP, args.lambda_child
        )
        va_loss, va_m = evaluate(
            model, va_loader, criterion_b, criterion_c, device,
            num_classes, PARENT_MAP, args.lambda_child
        )
        scheduler.step()

        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} "
              f"| val_acc={va_m['acc']:.4f} | val_f1={va_m['f1']:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
            w.writerow({
                "epoch": ep, "train_loss": tr_loss, "val_loss": va_loss,
                "val_acc": va_m["acc"], "val_f1": va_m["f1"],
                "test_acc": "", "test_f1": ""
            })

        if va_m["f1"] > best_val_f1:
            best_val_f1 = va_m["f1"]
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "lambda_child": args.lambda_child
            }, best_ckpt)
            print(f"** Saved best (val_f1={best_val_f1:.4f}) → {best_ckpt}")
            noimp = 0
        else:
            noimp += 1
            if patience > 0 and noimp >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs.")
                break

    # 测试
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"Loaded best checkpoint: {best_ckpt} (val_f1={best_val_f1:.4f})")

    te_loss, te_m = evaluate(
        model, te_loader, criterion_b, criterion_c, device,
        num_classes, PARENT_MAP, args.lambda_child
    )
    print(f"[TEST] loss={te_loss:.4f} | acc={te_m['acc']:.4f} | f1={te_m['f1']:.4f}")

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
        w.writerow({
            "epoch": "best", "train_loss": "", "val_loss": "",
            "val_acc": "", "val_f1": best_val_f1,
            "test_acc": te_m["acc"], "test_f1": te_m["f1"]
        })

    print(f"[LOG] saved → {log_path}")
    print(f"[CKPT] best → {best_ckpt}")

if __name__ == "__main__":
    main()
