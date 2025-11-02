import os, csv, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg

class NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)

class MLP2(nn.Module):
    def __init__(self, in_dim, hidden=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1)   # 输出1个logit
        )
    def forward(self, x): return self.net(x).squeeze(-1)  # [B]

def bin_metrics_from_probs(p, y, thr: float):
    yhat = (p >= thr).astype(np.int64)
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average='binary', zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "yhat": yhat}

def find_best_threshold(p, y, lo=0.20, hi=0.80, step=0.01):
    best_thr, best_f1, best_m = 0.5, -1.0, None
    thr = lo
    while thr <= hi + 1e-12:
        m = bin_metrics_from_probs(p, y, thr)
        if (m["f1"] > best_f1) or (abs(m["f1"] - best_f1) < 1e-12 and m["acc"] > (best_m["acc"] if best_m else -1)):
            best_thr, best_f1, best_m = thr, m["f1"], m
        thr += step
    return best_thr, best_f1, best_m

@torch.no_grad()
def infer_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x)                      # [B]
        p = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)

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
    ap.add_argument("--thr_lo", type=float, default=0.20)
    ap.add_argument("--thr_hi", type=float, default=0.80)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir",  type=str, default=cfg.LOG_DIR)
    ap.add_argument("--seed",     type=int, default=cfg.SEED)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    # 读取 CWT 紧凑特征与二分类标签
    Xtr = np.load(os.path.join(args.data_root, "Xcwt_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "Xcwt_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "Xcwt_test.npy"))
    ytr = np.load(os.path.join(args.data_root, "y_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, "y_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, "y_test.npy")).astype(np.int64)

    in_dim = int(Xtr.shape[1])
    tr_loader = DataLoader(NpyDataset(Xtr, ytr), batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(NpyDataset(Xva, yva), batch_size=args.batch_size_eval, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(NpyDataset(Xte, yte), batch_size=args.batch_size_eval, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    model = MLP2(in_dim=in_dim, hidden=args.hidden, p_drop=args.dropout).to(device)

    # 温和正类权重：w+ = sqrt(N_neg / N_pos)
    pos = max(int((ytr==1).sum()), 1); neg = int((ytr==0).sum())
    w_pos = (neg / pos) ** 0.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_pos], dtype=torch.float32, device=device))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)

    best_f1, best_state, best_thr = -1.0, None, 0.5
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_mlp_cwt_2cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_acc","val_f1","val_precision","val_recall","val_thr",
                                          "test_acc","test_f1","test_precision","test_recall","test_thr"])
        w.writeheader()

    no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tr_loader:
            x = x.to(device); y = y.to(device).float()
            optim.zero_grad()
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item() * y.size(0)
        sched.step()
        tr_loss = run_loss / max(len(tr_loader.dataset), 1)

        # 验证：扫描阈值取F1最优
        p_va, y_va = infer_probs(model, va_loader, device)
        thr, f1_opt, m_opt = find_best_threshold(p_va, y_va, lo=args.thr_lo, hi=args.thr_hi, step=args.thr_step)
        print(f"Epoch {ep:03d} | loss={tr_loss:.4f} | val thr={thr:.2f} acc={m_opt['acc']:.4f} "
              f"f1={m_opt['f1']:.4f} P={m_opt['precision']:.4f} R={m_opt['recall']:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_acc","val_f1","val_precision","val_recall","val_thr",
                                              "test_acc","test_f1","test_precision","test_recall","test_thr"])
            w.writerow({"epoch": ep, "train_loss": tr_loss, "val_acc": m_opt["acc"], "val_f1": m_opt["f1"],
                        "val_precision": m_opt["precision"], "val_recall": m_opt["recall"], "val_thr": thr,
                        "test_acc": "", "test_f1": "", "test_precision": "", "test_recall": "", "test_thr": ""})

        if f1_opt > best_f1:
            best_f1, best_thr = f1_opt, thr
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[EarlyStop] no improvement for {args.patience} epochs."); break

    # 测试：加载最佳&固定阈值
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    p_te, y_te = infer_probs(model, te_loader, device)
    m_te = bin_metrics_from_probs(p_te, y_te, best_thr)
    print(f"[TEST] thr={best_thr:.2f} | acc={m_te['acc']:.4f} f1={m_te['f1']:.4f} "
          f"P={m_te['precision']:.4f} R={m_te['recall']:.4f}")

    cm = confusion_matrix(y_te, (p_te >= best_thr).astype(np.int64), labels=[0,1])
    print(f"\n[TEST] Confusion Matrix [[TN,FP],[FN,TP]]:\n{cm}")

    ckpt_path = os.path.join(args.save_dir, f"mlp_cwt_2cls_best_{stamp}.pt")
    torch.save({"model": "MLP2_CWT", "state_dict": best_state, "best_thr": best_thr,
                "seed": args.seed, "in_dim": in_dim}, ckpt_path)

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_acc","val_f1","val_precision","val_recall","val_thr",
                                          "test_acc","test_f1","test_precision","test_recall","test_thr"])
        w.writerow({"epoch": "best", "train_loss": "", "val_acc": "", "val_f1": best_f1,
                    "val_precision": "", "val_recall": "", "val_thr": best_thr,
                    "test_acc": m_te["acc"], "test_f1": m_te["f1"],
                    "test_precision": m_te["precision"], "test_recall": m_te["recall"],
                    "test_thr": best_thr})
    print(f"[CKPT] {ckpt_path}\n[LOG]  {log_path}")

if __name__ == "__main__":
    main()