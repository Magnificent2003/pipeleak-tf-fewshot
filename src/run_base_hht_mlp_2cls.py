import os, csv, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import hilbert
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg

try:
    from PyEMD import CEEMDAN, EEMD
except Exception as e:
    CEEMDAN = None
    EEMD = None
    _PYEMD_IMPORT_ERR = e


class NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)


class MLP2(nn.Module):
    def __init__(self, in_dim, hidden=128, p_drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),  # 输出1个logit
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)  # [B]


def bin_metrics_from_probs(p, y, thr: float):
    yhat = (p >= thr).astype(np.int64)
    acc = accuracy_score(y, yhat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
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


def _build_hht_cache_key(args):
    key = (
        f"m{args.hht_method}_sr{args.sr}_tr{args.hht_trials}_nw{args.hht_noise_width}_"
        f"max{args.hht_max_imf}_keep{args.hht_keep_imf}_"
        f"fb{args.hht_freq_bins}_fmin{args.hht_fmin}_fmax{args.hht_fmax}"
    )
    return key.replace(".", "p")


def _make_decomposer(args):
    if args.hht_method == "ceemdan":
        if CEEMDAN is None:
            raise ImportError(
                f"PyEMD/CEEMDAN is required for HHT baseline. Original error: {_PYEMD_IMPORT_ERR}"
            )
        dec = CEEMDAN(trials=args.hht_trials, parallel=False)
        if hasattr(dec, "epsilon"):
            # 对应“noise width = 0.2 × signal std”的常用设置
            dec.epsilon = float(args.hht_noise_width)
        if hasattr(dec, "noise_scale"):
            dec.noise_scale = 1.0
        return dec

    if args.hht_method == "eemd":
        if EEMD is None:
            raise ImportError(
                f"PyEMD/EEMD is required for HHT baseline. Original error: {_PYEMD_IMPORT_ERR}"
            )
        dec = EEMD(trials=args.hht_trials, parallel=False)
        if hasattr(dec, "noise_width"):
            # EEMD 的噪声宽度（相对信号幅值）
            dec.noise_width = float(args.hht_noise_width)
        return dec

    raise ValueError(f"Unsupported hht_method: {args.hht_method}")


def hht_feature_1d(sig: np.ndarray, decomposer, args, bin_edges: np.ndarray):
    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    if np.std(x) < 1e-12:
        return np.zeros(args.hht_freq_bins + args.hht_keep_imf, dtype=np.float32)

    try:
        if args.hht_method == "ceemdan":
            try:
                imfs = decomposer.ceemdan(x, max_imf=args.hht_max_imf, progress=False)
            except TypeError:
                imfs = decomposer.ceemdan(x, max_imf=args.hht_max_imf)
        else:
            try:
                imfs = decomposer.eemd(x, max_imf=args.hht_max_imf, progress=False)
            except TypeError:
                imfs = decomposer.eemd(x, max_imf=args.hht_max_imf)
    except Exception:
        # 退化兜底：把原始信号当作单 IMF
        imfs = x[None, :]

    imfs = np.asarray(imfs, dtype=np.float64)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if imfs.shape[0] == 0:
        imfs = x[None, :]

    # 只保留前若干 IMF，不纳入 residual
    imfs = imfs[: args.hht_keep_imf]

    marginal = np.zeros(args.hht_freq_bins, dtype=np.float64)
    imf_energy = np.zeros(args.hht_keep_imf, dtype=np.float64)

    for k in range(imfs.shape[0]):
        imf = imfs[k]
        if np.allclose(imf, 0):
            continue

        z = hilbert(imf)
        amp = np.abs(z)
        phase = np.unwrap(np.angle(z))

        # 瞬时频率，长度 N-1；能量权重用对应时刻振幅平方
        inst_freq = np.diff(phase) * (args.sr / (2.0 * np.pi))
        weights = (amp[:-1] ** 2)

        valid = (
            np.isfinite(inst_freq)
            & np.isfinite(weights)
            & (inst_freq >= args.hht_fmin)
            & (inst_freq <= args.hht_fmax)
        )
        if np.any(valid):
            h, _ = np.histogram(inst_freq[valid], bins=bin_edges, weights=weights[valid])
            marginal += h

        imf_energy[k] = float(np.sum(imf ** 2))

    # marginal spectrum 归一化
    ms = float(np.sum(marginal))
    if ms > 0:
        marginal = marginal / (ms + 1e-12)

    # IMF 能量占比（前 keep_imf 个，不足补 0）
    es = float(np.sum(imf_energy))
    if es > 0:
        imf_ratio = imf_energy / (es + 1e-12)
    else:
        imf_ratio = np.zeros_like(imf_energy)

    feat = np.concatenate([marginal.astype(np.float32), imf_ratio.astype(np.float32)], axis=0)
    return feat


def extract_hht_split(X: np.ndarray, args, tag: str, seed_offset: int = 0):
    decomposer = _make_decomposer(args)
    bin_edges = np.linspace(args.hht_fmin, args.hht_fmax, args.hht_freq_bins + 1, dtype=np.float64)
    feats = []
    for i in range(X.shape[0]):
        if hasattr(decomposer, "noise_seed"):
            decomposer.noise_seed(int(args.seed + seed_offset + i))
        f = hht_feature_1d(X[i], decomposer, args, bin_edges)
        feats.append(f)
        if (i + 1) % 50 == 0 or (i + 1) == X.shape[0]:
            print(f"[HHT-{args.hht_method}-{tag}] {i+1}/{X.shape[0]} done")
    return np.vstack(feats).astype(np.float32)


def load_or_build_hht_features(Xtr, Xva, Xte, args):
    cache_dir = args.hht_cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _build_hht_cache_key(args)
        p_tr = os.path.join(cache_dir, f"Xhht_train_{key}.npy")
        p_va = os.path.join(cache_dir, f"Xhht_val_{key}.npy")
        p_te = os.path.join(cache_dir, f"Xhht_test_{key}.npy")
        if (not args.hht_force_rebuild) and all(os.path.exists(p) for p in [p_tr, p_va, p_te]):
            print(f"[HHT] Load cached features from {cache_dir}")
            return np.load(p_tr), np.load(p_va), np.load(p_te)

    print(f"[HHT] Extracting {args.hht_method.upper()} + Hilbert marginal spectrum features ...")
    Ftr = extract_hht_split(Xtr, args, tag="train", seed_offset=0)
    Fva = extract_hht_split(Xva, args, tag="val", seed_offset=100000)
    Fte = extract_hht_split(Xte, args, tag="test", seed_offset=200000)

    if cache_dir:
        key = _build_hht_cache_key(args)
        p_tr = os.path.join(cache_dir, f"Xhht_train_{key}.npy")
        p_va = os.path.join(cache_dir, f"Xhht_val_{key}.npy")
        p_te = os.path.join(cache_dir, f"Xhht_test_{key}.npy")
        np.save(p_tr, Ftr)
        np.save(p_va, Fva)
        np.save(p_te, Fte)
        print(f"[HHT] Saved cached features to {cache_dir}")

    return Ftr, Fva, Fte


@torch.no_grad()
def infer_probs(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x)
        p = torch.sigmoid(logit).detach().cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET)

    # HHT 参数（推荐配置）
    ap.add_argument("--sr", type=int, default=getattr(cfg, "FS", 8192))
    ap.add_argument("--hht_method", type=str, default="eemd", choices=["eemd", "ceemdan"])
    ap.add_argument("--hht_trials", type=int, default=20)
    ap.add_argument("--hht_noise_width", type=float, default=0.2)
    ap.add_argument("--hht_max_imf", type=int, default=6)
    ap.add_argument("--hht_keep_imf", type=int, default=5)
    ap.add_argument("--hht_freq_bins", type=int, default=64)
    ap.add_argument("--hht_fmin", type=float, default=0.0)
    ap.add_argument("--hht_fmax", type=float, default=-1.0)  # <=0 时自动取 sr/2

    # 可选缓存（HHT 提取较慢，默认开启）
    ap.add_argument("--hht_cache_dir", type=str, default=os.path.join(cfg.DATA_DIR, "dataset_hht"))
    ap.add_argument("--hht_force_rebuild", type=int, default=0)

    # MLP 训练参数
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
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.hht_fmax <= 0:
        args.hht_fmax = float(args.sr) / 2.0
    if args.hht_fmin < 0 or args.hht_fmin >= args.hht_fmax:
        raise ValueError(f"Invalid hht_fmin/hht_fmax: {args.hht_fmin}, {args.hht_fmax}")

    if args.hht_keep_imf <= 0 or args.hht_max_imf <= 0 or args.hht_keep_imf > args.hht_max_imf:
        raise ValueError("Require: 0 < hht_keep_imf <= hht_max_imf")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 读取原始 1D 信号与二分类标签
    Xtr = np.load(os.path.join(args.data_root, "X_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "X_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "X_test.npy"))
    ytr = np.load(os.path.join(args.data_root, "y_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, "y_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, "y_test.npy")).astype(np.int64)

    Ftr, Fva, Fte = load_or_build_hht_features(Xtr, Xva, Xte, args)

    # 用 train 统计做标准化
    mu = Ftr.mean(axis=0, keepdims=True)
    sg = Ftr.std(axis=0, keepdims=True) + 1e-6
    Ftr = (Ftr - mu) / sg
    Fva = (Fva - mu) / sg
    Fte = (Fte - mu) / sg

    in_dim = int(Ftr.shape[1])  # 默认 64 + 5 = 69
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

    model = MLP2(in_dim=in_dim, hidden=args.hidden, p_drop=args.dropout).to(device)

    # 温和正类权重：w+ = sqrt(N_neg / N_pos)
    pos = max(int((ytr == 1).sum()), 1)
    neg = int((ytr == 0).sum())
    w_pos = (neg / pos) ** 0.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w_pos], dtype=torch.float32, device=device))

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=20, gamma=0.5)

    best_f1, best_state, best_thr = -1.0, None, 0.5
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_mlp_hht_2cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "train_loss", "val_acc", "val_f1", "val_precision", "val_recall", "val_thr",
                "test_acc", "test_f1", "test_precision", "test_recall", "test_thr",
            ],
        )
        w.writeheader()

    no_improve = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tr_loader:
            x = x.to(device)
            y = y.to(device).float()
            optim.zero_grad()
            logit = model(x)
            loss = criterion(logit, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            run_loss += loss.item() * y.size(0)
        sched.step()
        tr_loss = run_loss / max(len(tr_loader.dataset), 1)

        # 验证：扫描阈值取 F1 最优
        p_va, y_va = infer_probs(model, va_loader, device)
        thr, f1_opt, m_opt = find_best_threshold(p_va, y_va, lo=args.thr_lo, hi=args.thr_hi, step=args.thr_step)
        print(
            f"Epoch {ep:03d} | loss={tr_loss:.4f} | val thr={thr:.2f} acc={m_opt['acc']:.4f} "
            f"f1={m_opt['f1']:.4f} P={m_opt['precision']:.4f} R={m_opt['recall']:.4f}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch", "train_loss", "val_acc", "val_f1", "val_precision", "val_recall", "val_thr",
                    "test_acc", "test_f1", "test_precision", "test_recall", "test_thr",
                ],
            )
            w.writerow(
                {
                    "epoch": ep,
                    "train_loss": tr_loss,
                    "val_acc": m_opt["acc"],
                    "val_f1": m_opt["f1"],
                    "val_precision": m_opt["precision"],
                    "val_recall": m_opt["recall"],
                    "val_thr": thr,
                    "test_acc": "",
                    "test_f1": "",
                    "test_precision": "",
                    "test_recall": "",
                    "test_thr": "",
                }
            )

        if f1_opt > best_f1:
            best_f1, best_thr = f1_opt, thr
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"[EarlyStop] no improvement for {args.patience} epochs.")
                break

    # 测试：加载最佳 & 固定阈值
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    p_te, y_te = infer_probs(model, te_loader, device)
    m_te = bin_metrics_from_probs(p_te, y_te, best_thr)
    print(
        f"[TEST] thr={best_thr:.2f} | acc={m_te['acc']:.4f} f1={m_te['f1']:.4f} "
        f"P={m_te['precision']:.4f} R={m_te['recall']:.4f}"
    )

    cm = confusion_matrix(y_te, (p_te >= best_thr).astype(np.int64), labels=[0, 1])
    print(f"\n[TEST] Confusion Matrix [[TN,FP],[FN,TP]]:\n{cm}")

    ckpt_path = os.path.join(args.save_dir, f"mlp_hht_2cls_best_{stamp}.pt")
    torch.save(
        {
            "model": "MLP2_HHT",
            "state_dict": best_state,
            "best_thr": best_thr,
            "seed": args.seed,
            "in_dim": in_dim,
            "hht_cfg": {
                "sr": args.sr,
                "hht_method": args.hht_method,
                "hht_trials": args.hht_trials,
                "hht_noise_width": args.hht_noise_width,
                "hht_max_imf": args.hht_max_imf,
                "hht_keep_imf": args.hht_keep_imf,
                "hht_freq_bins": args.hht_freq_bins,
                "hht_fmin": args.hht_fmin,
                "hht_fmax": args.hht_fmax,
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
                "epoch", "train_loss", "val_acc", "val_f1", "val_precision", "val_recall", "val_thr",
                "test_acc", "test_f1", "test_precision", "test_recall", "test_thr",
            ],
        )
        w.writerow(
            {
                "epoch": "best",
                "train_loss": "",
                "val_acc": "",
                "val_f1": best_f1,
                "val_precision": "",
                "val_recall": "",
                "val_thr": best_thr,
                "test_acc": m_te["acc"],
                "test_f1": m_te["f1"],
                "test_precision": m_te["precision"],
                "test_recall": m_te["recall"],
                "test_thr": best_thr,
            }
        )
    print(f"[CKPT] {ckpt_path}\n[LOG]  {log_path}")


if __name__ == "__main__":
    main()
