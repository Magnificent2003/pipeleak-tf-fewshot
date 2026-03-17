# run_base_stft_resnet_2cls.py
import os, csv, time, argparse, numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import config as cfg

# ---------- Dataset ----------
class NpyStftDataset(Dataset):
    """
    X: (N, 3, H, W) float32 in [0,1]
    y: (N,) int {0,1}
    统一使用 ImageNet 归一化，保证与 2D 主干对齐
    """
    def __init__(self, x_path, y_path):
        self.X = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path).astype(np.int64)
        assert self.X.ndim == 4 and self.X.shape[1] == 3, f"X shape must be (N,3,H,W), got {self.X.shape}"
        assert len(self.X) == len(self.y)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        x = (self.X[i] - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x))   # (3,H,W)
        y = torch.tensor(int(self.y[i]), dtype=torch.long)
        return x, y

# ---------- Metrics ----------
@torch.no_grad()
def metrics_from_logits_binary(logits: torch.Tensor, target: torch.Tensor):
    pred = logits.argmax(dim=1)
    acc = (pred == target).float().mean().item()
    # binary F1 (以 1 为正类)
    tp = ((pred==1)&(target==1)).sum().item()
    fp = ((pred==1)&(target==0)).sum().item()
    fn = ((pred==0)&(target==1)).sum().item()
    P  = tp / max(tp+fp, 1)
    R  = tp / max(tp+fn, 1)
    f1 = 2*P*R / max(P+R, 1e-12)
    return {"acc": float(acc), "f1": float(f1)}

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    run_loss = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * y.size(0)
    return run_loss / max(len(loader.dataset), 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    run_loss = 0.0
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        out = model(x)
        loss = criterion(out, y)
        run_loss += loss.item() * y.size(0)
        all_logits.append(out); all_y.append(y)
    n = max(len(loader.dataset), 1)
    logits = torch.cat(all_logits, 0); labels = torch.cat(all_y, 0)
    mets = metrics_from_logits_binary(logits, labels)
    return run_loss / n, mets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size",  type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs",     type=int, default=200)
    ap.add_argument("--lr",         type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers",  type=int, default=4)
    ap.add_argument("--save_dir",   type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir",    type=str, default=getattr(cfg, "LOG_DIR", "./logs"))
    ap.add_argument("--early_stop", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # paths
    Xtr = os.path.join(args.data_root, f"X_train_stft_{args.img_size}.npy")
    Xva = os.path.join(args.data_root, f"X_val_stft_{args.img_size}.npy")
    Xte = os.path.join(args.data_root, f"X_test_stft_{args.img_size}.npy")
    ytr = os.path.join(args.data_root, "y_train.npy")
    yva = os.path.join(args.data_root, "y_val.npy")
    yte = os.path.join(args.data_root, "y_test.npy")

    ds_tr = NpyStftDataset(Xtr, ytr); ds_va = NpyStftDataset(Xva, yva); ds_te = NpyStftDataset(Xte, yte)
    tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)
    va_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)
    te_loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers>0)

    num_classes = 2
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_stft_resnet18_2cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
        w.writeheader()

    best_val_f1, best_ckpt = -1.0, os.path.join(args.save_dir, f"stft_resnet18_2cls_best_{stamp}.pth")
    patience, noimp = args.early_stop, 0

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_m = evaluate(model, va_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} "
              f"| val_acc={va_m['acc']:.4f} | val_f1={va_m['f1']:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
            w.writerow({"epoch":ep, "train_loss":tr_loss, "val_loss":va_loss,
                        "val_acc":va_m["acc"], "val_f1":va_m["f1"], "test_acc":"", "test_f1":""})

        if va_m["f1"] > best_val_f1:
            best_val_f1 = va_m["f1"]
            torch.save({"state_dict": model.state_dict(), "num_classes": num_classes}, best_ckpt)
            print(f"** Saved best (val_f1={best_val_f1:.4f}) → {best_ckpt}")
            noimp = 0
        else:
            noimp += 1
            if patience > 0 and noimp >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs."); break

    # Test
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"Loaded best checkpoint: {best_ckpt} (val_f1={best_val_f1:.4f})")

    te_loss, te_m = evaluate(model, te_loader, criterion, device)
    print(f"[TEST] loss={te_loss:.4f} | acc={te_m['acc']:.4f} | f1={te_m['f1']:.4f}")

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
        w.writerow({"epoch":"best", "train_loss":"", "val_loss":"", "val_acc":"", "val_f1":best_val_f1,
                    "test_acc":te_m["acc"], "test_f1":te_m["f1"]})

    print(f"[LOG] saved → {log_path}")
    print(f"[CKPT] best → {best_ckpt}")

if __name__ == "__main__":
    main()
