import os, csv, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg

# ===== Dataset (CWT 紧凑特征) =====
class NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)

# ===== 小型 MLP（单隐层） =====
class MLP4(nn.Module):
    def __init__(self, in_dim, hidden=128, p_drop=0.2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes)   # logits: [B, C]
        )
    def forward(self, x): return self.net(x)

def mc_metrics_from_logits(logits: np.ndarray, y: np.ndarray):
    yhat = logits.argmax(axis=1)
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "yhat": yhat}

@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval()
    logits_list, labels = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        logits_list.append(out.detach().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits_list), np.concatenate(labels)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_CWT)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--batch_size_eval", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir",  type=str, default=cfg.LOG_DIR)
    ap.add_argument("--seed",     type=int, default=cfg.SEED)
    args = ap.parse_args()

    # ---- seeds & device ----
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    # ---- load data ----
    Xtr = np.load(os.path.join(args.data_root, "Xcwt_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "Xcwt_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "Xcwt_test.npy"))
    ytr = np.load(os.path.join(args.data_root, "y4_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, "y4_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, "y4_test.npy")).astype(np.int64)

    in_dim = int(Xtr.shape[1]); num_classes = 4
    tr_loader = DataLoader(NpyDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(NpyDataset(Xva, yva), batch_size=args.batch_size_eval, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(NpyDataset(Xte, yte), batch_size=args.batch_size_eval, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    # ---- model / loss / opt ----
    model = MLP4(in_dim=in_dim, hidden=args.hidden, p_drop=args.dropout, num_classes=num_classes).to(device)
    # 类不均衡：温和权重（开平方反频率）
    counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = counts[counts > 0].min() if (counts > 0).any() else 1.0
    class_w = (counts.max() / counts) ** 0.5
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)

    # ---- training ----
    best_f1, best_state = -1.0, None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_mlp4_cwt_4cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "epoch","train_loss","val_acc","val_macro_f1","val_precision_macro","val_recall_macro",
            "test_acc","test_macro_f1"
        ])
        w.writeheader()

    no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tr_loader:
            x = x.to(device); y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item() * y.size(0)
        sched.step()

        # val
        logits_va, y_va = eval_logits(model, va_loader, device)
        m_va = mc_metrics_from_logits(logits_va, y_va)
        tr_loss = run_loss / max(len(tr_loader.dataset), 1)
        print(f"Epoch {ep:03d} | loss={tr_loss:.4f} | val acc={m_va['acc']:.4f} f1(Macro)={m_va['f1']:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=[
                "epoch","train_loss","val_acc","val_macro_f1","val_precision_macro","val_recall_macro",
                "test_acc","test_macro_f1"
            ])
            w.writerow({
                "epoch": ep, "train_loss": float(tr_loss),
                "val_acc": m_va["acc"], "val_macro_f1": m_va["f1"],
                "val_precision_macro": m_va["precision"], "val_recall_macro": m_va["recall"],
                "test_acc": "", "test_macro_f1": ""
            })

        if m_va["f1"] > best_f1:
            best_f1 = m_va["f1"]; best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[EarlyStop] no improvement for {args.patience} epochs."); break

    # ---- test ----
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    logits_te, y_te = eval_logits(model, te_loader, device)
    m_te = mc_metrics_from_logits(logits_te, y_te)
    print(f"[TEST] acc={m_te['acc']:.4f} macro-F1={m_te['f1']:.4f}")

    # 混淆矩阵打印（可注释）
    cm = confusion_matrix(y_te, m_te["yhat"], labels=list(range(num_classes)))
    print(f"\n[TEST] Confusion Matrix (rows=true, cols=pred):\n{cm}")

    # save ckpt + final row
    ckpt_path = os.path.join(args.save_dir, f"mlp4_cwt_best_{stamp}.pt")
    torch.save({"model": "MLP4_CWT", "state_dict": best_state, "num_classes": num_classes, "seed": args.seed}, ckpt_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "epoch","train_loss","val_acc","val_macro_f1","val_precision_macro","val_recall_macro",
            "test_acc","test_macro_f1"
        ])
        w.writerow({
            "epoch": "best", "train_loss": "", "val_acc": "", "val_macro_f1": best_f1,
            "val_precision_macro": "", "val_recall_macro": "",
            "test_acc": m_te["acc"], "test_macro_f1": m_te["f1"]
        })
    print(f"[CKPT] {ckpt_path}\n[LOG]  {log_path}")

if __name__ == "__main__":
    main()
