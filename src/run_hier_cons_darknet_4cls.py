import os, csv, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F   # 新增
import torch.optim as optim
import config as cfg

from typing import Dict
from torch.utils.data import DataLoader
from NpyDataset import NpyDataset
from sklearn.metrics import f1_score
from HierarchicalDarknet19 import HierarchicalDarknet19

# ====== HCL 固定超参（非开关）======
W_CONSIST = 0.1      # 一致性损失系数（对称 KL）
EPS = 1e-8           # 数值稳定
TAU = 2.0            # 温度系数

# ---------- metrics ----------
@torch.no_grad()
def metrics_from_logits(logits: torch.Tensor, y: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    acc = float((pred == y).float().mean().item())
    y_np = y.cpu().numpy()
    p_np = pred.cpu().numpy()
    avg = "binary" if num_classes == 2 else "macro"
    f1 = float(f1_score(y_np, p_np, average=avg, zero_division=0))
    return {"acc": acc, "f1": f1}

# ---------- train & eval with HCL ----------
def train_one_epoch(model, loader, criterion_b, criterion_c, optimizer, device, parent_map, lambda_child, M):
    model.train()
    loss_sum, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long)
        y_parent = parent_map[y]

        optimizer.zero_grad(set_to_none=True)
        logits_b, logits_c = model(x)  # 父头(2类)、子头(4类)

        # 监督项
        loss_sup = criterion_b(logits_b, y_parent) + lambda_child * criterion_c(logits_c, y)

        # 一致性项（对称 KL）
        pb_log = F.log_softmax(logits_b, dim=1)    # [B,2]
        pb     = pb_log.exp()                      # [B,2]
        pc     = F.softmax(logits_c, dim=1)        # [B,4]
        pb_hat = torch.matmul(pc, M.t())           # [B,2] = 子类概率聚合到父级

        kl_b2c = F.kl_div(pb_log, pb_hat, reduction="batchmean")              # KL(p_b || pb_hat)
        kl_c2b = F.kl_div(torch.log(pb_hat + EPS), pb, reduction="batchmean") # KL(pb_hat || p_b)
        loss_cons = kl_b2c + kl_c2b

        loss = loss_sup + W_CONSIST * loss_cons
        loss.backward()
        optimizer.step()

        loss_sum += float(loss.item()) * x.size(0)
        n += x.size(0)
    return loss_sum / max(n, 1)

@torch.no_grad()
def evaluate(model, loader, criterion_b, criterion_c, device, num_classes, parent_map, lambda_child, M):
    model.eval()
    loss_sum, n = 0.0, 0
    all_logits_c, all_logits_b, all_y = [], [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.long)
        y_parent = parent_map[y]

        logits_b, logits_c = model(x)

        # 监督项
        loss_sup = criterion_b(logits_b, y_parent) + lambda_child * criterion_c(logits_c, y)

        # 一致性项（对称 KL）
        pc     = F.softmax(logits_c, dim=1)
        pb_hat = torch.matmul(pc, M.t())
        loss_cons = F.nll_loss(torch.log(pb_hat + 1e-8), y_parent, reduction="mean")
        loss = loss_sup - W_CONSIST * loss_cons
        loss_sum += float(loss.item()) * x.size(0); n += x.size(0)

        all_logits_b.append(logits_b)
        all_logits_c.append(logits_c)
        all_y.append(y)

    logits_b = torch.cat(all_logits_b, dim=0)  # [N,2]
    logits_c = torch.cat(all_logits_c, dim=0)  # [N,4]
    ys       = torch.cat(all_y,        dim=0)  # [N]

    # —— 一致性解码：父先验掩蔽子类，再计算子头指标 ——
    parent_pred = logits_b.argmax(dim=1)                              # [N]
    allow = (parent_map.view(1, -1) == parent_pred.view(-1, 1))       # [N,4] bool
    mask  = torch.where(allow, torch.zeros_like(logits_c), torch.full_like(logits_c, -1e9))
    logits_c_masked = logits_c + mask

    mets = metrics_from_logits(logits_c_masked, ys, num_classes)
    return loss_sum / max(n, 1), mets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--label_prefix", type=str, default="y4")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--early_stop", type=int, default=0)
    ap.add_argument("--lambda_child", type=float, default=0.3)
    args = ap.parse_args()

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

    num_classes = int(args.num_classes)
    for split_name, ds in [("train", ds_tr), ("val", ds_va), ("test", ds_te)]:
        y_min, y_max = int(np.min(ds.y)), int(np.max(ds.y))
        if y_min < 0 or y_max >= num_classes:
            print(f"[WARN] {split_name} 标签范围 {y_min}..{y_max} 超出 [0,{num_classes-1}]")

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
    parent_map = torch.tensor([1, 1, 0, 0], dtype=torch.long, device=device)  # idx 0,1,2,3

    # 构造聚合矩阵 M ∈ R^{2×4}（子→父）
    M = torch.zeros(2, num_classes, dtype=torch.float32, device=device)
    for k in range(num_classes):
        M[parent_map[k], k] = 1.0
    M.requires_grad_(False)

    # 日志
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_darknet19_hcl_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
        w.writeheader()

    best_val_f1 = -1.0
    best_ckpt = os.path.join(args.save_dir, f"darknet19_hcl_best_{stamp}.pth")
    patience, noimp = args.early_stop, 0

    # 训练
    for ep in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(
            model, tr_loader, criterion_b, criterion_c, optimizer, device,
            parent_map, args.lambda_child, M
        )
        va_loss, va_m = evaluate(
            model, va_loader, criterion_b, criterion_c, device,
            num_classes, parent_map, args.lambda_child, M
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
                "lambda_child": args.lambda_child,
                "w_consist": W_CONSIST
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
        num_classes, parent_map, args.lambda_child, M
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
