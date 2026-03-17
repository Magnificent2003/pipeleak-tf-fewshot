import os, csv, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg


class NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)


class MLP4(nn.Module):
    def __init__(self, in_dim, hidden=128, p_drop=0.2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)  # [B, C]


def mc_metrics_from_logits(logits: np.ndarray, y: np.ndarray):
    yhat = logits.argmax(axis=1)
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="macro", zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1), "yhat": yhat}


def parent_metrics_from_cm4(cm4: np.ndarray):
    # Parent mapping: {0,1}->0 and {2,3}->1
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
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0).astype(np.float32)  # [2*n_mfcc]
    return feat


def extract_mfcc_stats_split(X: np.ndarray, args, tag: str):
    feats = []
    for i in range(X.shape[0]):
        f = mfcc_stats_1d(
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
        feats.append(f)
        if (i + 1) % 200 == 0 or (i + 1) == X.shape[0]:
            print(f"[MFCC-{tag}] {i+1}/{X.shape[0]} done")
    return np.vstack(feats).astype(np.float32)


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
    ap.add_argument("--data_root", type=str, default=cfg.DATASET)
    ap.add_argument("--label_prefix", type=str, default="y4")

    # MFCC 参数
    ap.add_argument("--sr", type=int, default=getattr(cfg, "FS", 8192))
    ap.add_argument("--n_fft", type=int, default=256)
    ap.add_argument("--win_length", type=int, default=256)
    ap.add_argument("--hop_length", type=int, default=128)
    ap.add_argument("--window", type=str, default="hamming")
    ap.add_argument("--n_mels", type=int, default=40)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=-1.0)  # <=0 时自动取 sr/2
    ap.add_argument("--center", type=int, default=1)     # 1=True, 0=False

    # MLP 参数
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
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.fmax <= 0:
        args.fmax = float(args.sr) / 2.0
    if args.fmin < 0 or args.fmin >= args.fmax:
        raise ValueError(f"Invalid fmin/fmax: fmin={args.fmin}, fmax={args.fmax}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 读取原始 1D 信号与 4cls 标签
    Xtr = np.load(os.path.join(args.data_root, "X_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "X_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "X_test.npy"))
    ytr = np.load(os.path.join(args.data_root, f"{args.label_prefix}_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, f"{args.label_prefix}_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, f"{args.label_prefix}_test.npy")).astype(np.int64)

    print("Extracting MFCC mean+std features ...")
    Ftr = extract_mfcc_stats_split(Xtr, args, tag="train")
    Fva = extract_mfcc_stats_split(Xva, args, tag="val")
    Fte = extract_mfcc_stats_split(Xte, args, tag="test")

    # 用 train 统计做标准化
    mu = Ftr.mean(axis=0, keepdims=True)
    sg = Ftr.std(axis=0, keepdims=True) + 1e-6
    Ftr = (Ftr - mu) / sg
    Fva = (Fva - mu) / sg
    Fte = (Fte - mu) / sg

    num_classes = int(np.max(ytr)) + 1
    tr_loader = DataLoader(
        NpyDataset(Ftr, ytr), batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    va_loader = DataLoader(
        NpyDataset(Fva, yva), batch_size=args.batch_size_eval, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    te_loader = DataLoader(
        NpyDataset(Fte, yte), batch_size=args.batch_size_eval, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    model = MLP4(in_dim=Ftr.shape[1], hidden=args.hidden, p_drop=args.dropout, num_classes=num_classes).to(device)

    # 类不均衡：温和权重（开平方反频率）
    counts = np.bincount(ytr, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = counts[counts > 0].min() if (counts > 0).any() else 1.0
    class_w = (counts.max() / counts) ** 0.5
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)

    best_f1, best_state = -1.0, None
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_mlp_mfcc_4cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "val_acc", "val_macro_f1", "val_precision_macro", "val_recall_macro",
                "test_acc", "test_macro_f1", "test_recall_macro", "test_parent_f1", "test_parent_recall",
            ],
        )
        w.writeheader()

    no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tr_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item() * y.size(0)
        sched.step()

        logits_va, y_va = eval_logits(model, va_loader, device)
        m_va = mc_metrics_from_logits(logits_va, y_va)
        tr_loss = run_loss / max(len(tr_loader.dataset), 1)
        print(f"Epoch {ep:03d} | loss={tr_loss:.4f} | val acc={m_va['acc']:.4f} f1(Macro)={m_va['f1']:.4f}")

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch", "train_loss", "val_acc", "val_macro_f1", "val_precision_macro", "val_recall_macro",
                    "test_acc", "test_macro_f1", "test_recall_macro", "test_parent_f1", "test_parent_recall",
                ],
            )
            w.writerow(
                {
                    "epoch": ep,
                    "train_loss": float(tr_loss),
                    "val_acc": m_va["acc"],
                    "val_macro_f1": m_va["f1"],
                    "val_precision_macro": m_va["precision"],
                    "val_recall_macro": m_va["recall"],
                    "test_acc": "",
                    "test_macro_f1": "",
                    "test_recall_macro": "",
                    "test_parent_f1": "",
                    "test_parent_recall": "",
                }
            )

        if m_va["f1"] > best_f1:
            best_f1 = m_va["f1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[EarlyStop] no improvement for {args.patience} epochs.")
                break

    # test
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    logits_te, y_te = eval_logits(model, te_loader, device)
    m_te = mc_metrics_from_logits(logits_te, y_te)
    print(f"[TEST] acc={m_te['acc']:.4f} macro-F1={m_te['f1']:.4f} macro-Recall={m_te['recall']:.4f}")

    cm = confusion_matrix(y_te, m_te["yhat"], labels=list(range(num_classes)))
    print(f"\n[TEST] Confusion Matrix (rows=true, cols=pred):\n{cm}")
    cm_parent, parent_f1_te, parent_rec_te = parent_metrics_from_cm4(cm)
    print(f"\n[TEST] Parent Confusion Matrix (rows=true, cols=pred, 01->0,23->1):\n{cm_parent}")
    print(f"[TEST] 4-class Parent-F1={parent_f1_te:.4f} Parent-Recall={parent_rec_te:.4f}")

    ckpt_path = os.path.join(args.save_dir, f"mlp_mfcc_4cls_best_{stamp}.pt")
    torch.save(
        {
            "model": "MLP4_MFCC",
            "state_dict": best_state,
            "num_classes": num_classes,
            "seed": args.seed,
            "mfcc_cfg": {
                "sr": args.sr,
                "n_fft": args.n_fft,
                "win_length": args.win_length,
                "hop_length": args.hop_length,
                "window": args.window,
                "n_mels": args.n_mels,
                "n_mfcc": args.n_mfcc,
                "fmin": args.fmin,
                "fmax": args.fmax,
                "center": bool(args.center),
            },
            "feat_mu": mu.squeeze(0).astype(np.float32),
            "feat_sigma": sg.squeeze(0).astype(np.float32),
        },
        ckpt_path,
    )

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "val_acc", "val_macro_f1", "val_precision_macro", "val_recall_macro",
                "test_acc", "test_macro_f1", "test_recall_macro", "test_parent_f1", "test_parent_recall",
            ],
        )
        w.writerow(
            {
                "epoch": "best",
                "train_loss": "",
                "val_acc": "",
                "val_macro_f1": best_f1,
                "val_precision_macro": "",
                "val_recall_macro": "",
                "test_acc": m_te["acc"],
                "test_macro_f1": m_te["f1"],
                "test_recall_macro": m_te["recall"],
                "test_parent_f1": parent_f1_te,
                "test_parent_recall": parent_rec_te,
            }
        )
    print(f"[CKPT] {ckpt_path}\n[LOG]  {log_path}")


if __name__ == "__main__":
    main()
