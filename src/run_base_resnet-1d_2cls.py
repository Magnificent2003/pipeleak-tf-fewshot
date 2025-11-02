import os, csv, argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import config as cfg

class SignalsDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mu: float, sigma: float):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mu = float(mu)
        self.sigma = float(sigma) if sigma > 0 else 1.0
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = (self.X[i] - self.mu) / self.sigma     # [L]
        x = torch.from_numpy(x)[None, ...]         # [1, L]
        y = torch.tensor(int(self.y[i]), dtype=torch.long)
        return x, y

# ===== ResNet-1D (18-layer) =====
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm1d(planes)
        self.down  = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.down = nn.Sequential(
                nn.Conv1d(in_planes, planes*self.expansion, 1, stride, bias=False),
                nn.BatchNorm1d(planes*self.expansion),
            )
    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            idt = self.down(x)
        out = self.relu(out + idt)
        return out

class ResNet1D(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv1d(1, 64, 7, 2, 3, bias=False)
        self.bn1   = nn.BatchNorm1d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(3, 2, 1)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Linear(512, num_classes)     # 2 类输出
    def _make_layer(self, planes, blocks, stride):
        layers = [BasicBlock1D(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock1D(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.gap(x).squeeze(-1)
        logits = self.fc(x)                        # [B,2]
        return logits

# ===== Train / Eval =====
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    run_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        run_loss += loss.item() * y.size(0)
    return run_loss / max(len(loader.dataset), 1)

@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=2):
    model.eval()
    total, correct, run_loss = 0, 0, 0.0
    y_true, y_pred = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss   = criterion(logits, y)
        run_loss += loss.item() * y.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(pred.detach().cpu().numpy())
    acc = correct / max(total, 1)
    if y_true:
        yt = np.concatenate(y_true)
        yp = np.concatenate(y_pred)
        tp = int(((yt==1)&(yp==1)).sum()); fp = int(((yt==0)&(yp==1)).sum()); fn = int(((yt==1)&(yp==0)).sum())
        P  = tp / max(tp+fp, 1); R = tp / max(tp+fn, 1)
        f1 = 2*P*R / max(P+R, 1e-12)
    else:
        f1 = 0.0
    return run_loss / max(len(loader.dataset), 1), {"acc": acc, "f1": f1}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir",  type=str, default=cfg.LOG_DIR)
    ap.add_argument("--early_stop", type=int, default=0)  # 0=关闭
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir,  exist_ok=True)

    # 路径
    Xtr = os.path.join(args.data_root, "X_train.npy")
    Xva = os.path.join(args.data_root, "X_val.npy")
    Xte = os.path.join(args.data_root, "X_test.npy")
    ytr = os.path.join(args.data_root, "y_train.npy")
    yva = os.path.join(args.data_root, "y_val.npy")
    yte = os.path.join(args.data_root, "y_test.npy")

    # 载入与标准化统计（仅由 train 估计）
    X_train = np.load(Xtr); X_val = np.load(Xva); X_test = np.load(Xte)
    y_train = np.load(ytr); y_val = np.load(yva); y_test = np.load(yte)
    mu, sigma = float(X_train.mean()), float(X_train.std() + 1e-12)

    ds_tr = SignalsDataset(X_train, y_train, mu, sigma)
    ds_va = SignalsDataset(X_val,   y_val,   mu, sigma)
    ds_te = SignalsDataset(X_test,  y_test,  mu, sigma)

    tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, pin_memory=True)
    va_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    te_loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    num_classes = 2
    model = ResNet1D(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_resnet1d_2cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
        w.writeheader()

    best_val_f1, best_ckpt = -1.0, os.path.join(args.save_dir, f"resnet1d_2cls_best_{stamp}.pth")
    patience, noimp = args.early_stop, 0

    for ep in range(1, args.epochs+1):
        tr_loss = train_one_epoch(model, tr_loader, criterion, optimizer, device)
        va_loss, va_m = evaluate(model, va_loader, criterion, device, num_classes)
        scheduler.step()

        print(f"Epoch {ep:03d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} "
              f"| val_acc={va_m['acc']:.4f} | val_f1={va_m['f1']:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
            w.writerow({"epoch":ep, "train_loss":tr_loss, "val_loss":va_loss,
                        "val_acc":va_m["acc"], "val_f1":va_m["f1"], "test_acc":"", "test_f1":""})

        if va_m["f1"] > best_val_f1:
            best_val_f1 = va_m["f1"]
            torch.save({"state_dict": model.state_dict(), "num_classes": num_classes,
                        "mu": mu, "sigma": sigma}, best_ckpt)
            print(f"** Saved best (val_f1={best_val_f1:.4f}) → {best_ckpt}")
            noimp = 0
        else:
            noimp += 1
            if patience > 0 and noimp >= patience:
                print(f"[EarlyStop] no improvement for {patience} epochs.")
                break

    # 测试集评估
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print(f"Loaded best checkpoint: {best_ckpt} (val_f1={best_val_f1:.4f})")

    te_loss, te_m = evaluate(model, te_loader, criterion, device, num_classes)
    print(f"[TEST] loss={te_loss:.4f} | acc={te_m['acc']:.4f} | f1={te_m['f1']:.4f}")

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch","train_loss","val_loss","val_acc","val_f1","test_acc","test_f1"])
        w.writerow({"epoch":"best", "train_loss":"", "val_loss":"", "val_acc":"", "val_f1":best_val_f1,
                    "test_acc":te_m["acc"], "test_f1":te_m["f1"]})

    print(f"[LOG] saved → {log_path}")
    print(f"[CKPT] best → {best_ckpt}")

if __name__ == "__main__":
    main()
