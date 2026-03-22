import argparse
import csv
import glob
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

import config as cfg
from Darknet19 import Darknet19
from NpyDataset import NpyDataset
from fusion_2cls_models import CWGFFusion2Cls


class MLP2(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, p_drop: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def resolve_latest_ckpt(path_arg: str, pattern: str, desc: str) -> str:
    if path_arg.strip():
        if not os.path.exists(path_arg):
            raise FileNotFoundError(f"{desc} checkpoint not found: {path_arg}")
        return path_arg
    cands = glob.glob(os.path.join(cfg.CKPT_DIR, pattern))
    if not cands:
        raise FileNotFoundError(
            f"Cannot auto-find {desc} checkpoint by pattern '{pattern}' under {cfg.CKPT_DIR}. "
            f"Please pass --ckpt_{desc}."
        )
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def bin_metrics_from_probs(p: np.ndarray, y: np.ndarray, thr: float):
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: p={p.shape[0]}, y={y.shape[0]}")
    y_hat = (p >= thr).astype(np.int64)
    cm = confusion_matrix(y, y_hat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary", zero_division=0)
    bal_acc = 0.5 * ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1)))
    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "bal_acc": float(bal_acc),
        "cm": cm,
    }


def find_best_thr(p: np.ndarray, y: np.ndarray, lo: float, hi: float, step: float, objective: str = "f1"):
    n = int(np.floor((hi - lo) / step)) + 1
    taus = np.round(np.linspace(lo, hi, n), 6)

    def _score(tau: float) -> float:
        mets = bin_metrics_from_probs(p, y, tau)
        return mets["f1"] if objective == "f1" else mets["acc"]

    scores = np.array([_score(float(t)) for t in taus], dtype=np.float64)
    i = int(np.argmax(scores))
    tau1 = float(taus[i])

    lo2 = max(lo, tau1 - 5 * step)
    hi2 = min(hi, tau1 + 5 * step)
    step2 = step / 10.0
    n2 = int(np.floor((hi2 - lo2) / step2)) + 1
    taus2 = np.round(np.linspace(lo2, hi2, n2), 6)
    scores2 = np.array([_score(float(t)) for t in taus2], dtype=np.float64)
    j = int(np.argmax(scores2))
    return float(taus2[j]), float(scores2[j])


def bce_on_probs(p: torch.Tensor, y: torch.Tensor, w_pos: float = 1.0, w_neg: float = 1.0, eps: float = 1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return (-w_pos * y * torch.log(p) - w_neg * (1.0 - y) * torch.log(1.0 - p)).mean()


def entropy_pos(p: torch.Tensor, eps: float = 1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return (-(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))).mean()


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def load_darknet_2cls(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))
    num_classes = int(ck.get("num_classes", 2))
    model = Darknet19(num_classes=num_classes)
    model.load_state_dict(state, strict=False)
    model.to(device)
    freeze_model(model)
    return model


def _find_linear_dims_from_state_dict(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    w1 = state.get("net.0.weight", None)
    w2 = state.get("net.3.weight", None)
    if w1 is None or w2 is None:
        raise KeyError("Cannot infer MLP2 dims from checkpoint state_dict (need net.0.weight and net.3.weight).")
    hidden = int(w1.shape[0])
    in_dim = int(w1.shape[1])
    if int(w2.shape[1]) != hidden:
        raise ValueError(f"Unexpected MLP2 shape: net.0={tuple(w1.shape)}, net.3={tuple(w2.shape)}")
    return in_dim, hidden


def load_mfcc_mlp_2cls(
    ckpt_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, float], np.ndarray, np.ndarray]:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))

    if "in_dim" in ck:
        in_dim = int(ck["in_dim"])
        hidden = int(state["net.0.weight"].shape[0])
    else:
        in_dim, hidden = _find_linear_dims_from_state_dict(state)

    model = MLP2(in_dim=in_dim, hidden=hidden, p_drop=0.0)
    model.load_state_dict(state, strict=False)
    model.to(device)
    freeze_model(model)

    mfcc_cfg = dict(ck.get("mfcc_cfg", {}))

    feat_mu = ck.get("feat_mu", None)
    feat_sigma = ck.get("feat_sigma", None)
    if feat_mu is not None and feat_sigma is not None:
        mu = np.asarray(feat_mu, dtype=np.float32).reshape(1, -1)
        sg = np.asarray(feat_sigma, dtype=np.float32).reshape(1, -1)
    else:
        mu = None
        sg = None
    return model, mfcc_cfg, mu, sg


@torch.no_grad()
def logits_from_darknet(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if out.ndim != 2 or out.shape[1] != 2:
            raise ValueError(f"Expected [B,2] logits from DarkNet, got {tuple(out.shape)}")
        logits_all.append(out.detach().cpu().numpy().astype(np.float32))
        y_all.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


@torch.no_grad()
def logits_from_mlp(model: nn.Module, feats: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    n = feats.shape[0]
    for i in range(0, n, batch_size):
        x = torch.from_numpy(feats[i : i + batch_size]).to(device)
        logit_pos = model(x)  # [B]
        logits = torch.stack([-logit_pos, logit_pos], dim=1)  # [B,2]
        outs.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


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
) -> np.ndarray:
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


def extract_mfcc_stats_split(X: np.ndarray, mfcc_cfg: Dict[str, float], tag: str) -> np.ndarray:
    feats = []
    n = X.shape[0]
    for i in range(n):
        f = mfcc_stats_1d(
            X[i],
            sr=int(mfcc_cfg["sr"]),
            n_fft=int(mfcc_cfg["n_fft"]),
            win_length=int(mfcc_cfg["win_length"]),
            hop_length=int(mfcc_cfg["hop_length"]),
            window=str(mfcc_cfg["window"]),
            n_mels=int(mfcc_cfg["n_mels"]),
            n_mfcc=int(mfcc_cfg["n_mfcc"]),
            fmin=float(mfcc_cfg["fmin"]),
            fmax=float(mfcc_cfg["fmax"]),
            center=bool(mfcc_cfg["center"]),
        )
        feats.append(f)
        if (i + 1) % 200 == 0 or (i + 1) == n:
            print(f"[MFCC-{tag}] {i + 1}/{n} done")
    return np.vstack(feats).astype(np.float32)


def build_stft_loaders(args):
    root = args.data_root_stft
    x_train = os.path.join(root, f"X_train_stft_{args.img_size}.npy")
    x_val = os.path.join(root, f"X_val_stft_{args.img_size}.npy")
    x_test = os.path.join(root, f"X_test_stft_{args.img_size}.npy")
    y_train = os.path.join(root, "y_train.npy")
    y_val = os.path.join(root, "y_val.npy")
    y_test = os.path.join(root, "y_test.npy")

    required = [x_train, x_val, x_test, y_train, y_val, y_test]
    miss = [p for p in required if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("Missing STFT dataset files:\n" + "\n".join(miss))

    ds_tr = NpyDataset(x_train, y_train, normalize="imagenet", memmap=True)
    ds_va = NpyDataset(x_val, y_val, normalize="imagenet", memmap=True)
    ds_te = NpyDataset(x_test, y_test, normalize="imagenet", memmap=True)

    dl_tr = torch.utils.data.DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dl_te = torch.utils.data.DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return dl_tr, dl_va, dl_te


def load_raw_splits(data_root_raw: str):
    paths = {
        "X_train": os.path.join(data_root_raw, "X_train.npy"),
        "X_val": os.path.join(data_root_raw, "X_val.npy"),
        "X_test": os.path.join(data_root_raw, "X_test.npy"),
        "y_train": os.path.join(data_root_raw, "y_train.npy"),
        "y_val": os.path.join(data_root_raw, "y_val.npy"),
        "y_test": os.path.join(data_root_raw, "y_test.npy"),
    }
    miss = [p for p in paths.values() if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("Missing raw dataset files:\n" + "\n".join(miss))

    out = {k: np.load(v) for k, v in paths.items()}
    out["y_train"] = out["y_train"].astype(np.int64)
    out["y_val"] = out["y_val"].astype(np.int64)
    out["y_test"] = out["y_test"].astype(np.int64)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_raw", type=str, default=cfg.DATASET)
    ap.add_argument("--data_root_stft", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt_mfcc2", type=str, default="")
    ap.add_argument("--ckpt_darknet2", type=str, default="")

    ap.add_argument("--fusion_hid", type=int, default=16)
    ap.add_argument("--fusion_drop", type=float, default=0.1)
    ap.add_argument("--fusion_epochs", type=int, default=200)
    ap.add_argument("--fusion_lr", type=float, default=1e-4)
    ap.add_argument("--fusion_wd", type=float, default=1e-3)
    ap.add_argument("--lambda_ent", type=float, default=0.1)
    ap.add_argument("--lambda_center", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=20)

    ap.add_argument("--thr_lo", type=float, default=0.20)
    ap.add_argument("--thr_hi", type=float, default=0.80)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--thr_objective", type=str, default="f1", choices=["f1", "acc"])
    ap.add_argument("--pos_pow", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=int(getattr(cfg, "SEED", 2023)))

    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--save_ckpt", type=int, default=0)
    return ap


def run_fusion(args) -> None:
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    if bool(args.save_ckpt):
        os.makedirs(args.save_dir, exist_ok=True)

    ckpt_mfcc = resolve_latest_ckpt(args.ckpt_mfcc2, "mlp_mfcc_2cls_best_*.pt", "mfcc2")
    ckpt_dark = resolve_latest_ckpt(args.ckpt_darknet2, "darknet19_best_*.pth", "darknet2")

    print("========== Fuse MFCC+DarkNet (2cls, CWGF) ==========")
    print(f"device       : {device}")
    print(f"data_root_raw: {args.data_root_raw}")
    print(f"data_root_stft: {args.data_root_stft}")
    print(f"ckpt_mfcc2   : {ckpt_mfcc}")
    print(f"ckpt_darknet2: {ckpt_dark}")
    print("====================================================")

    model_dark = load_darknet_2cls(ckpt_dark, device)
    model_mfcc, mfcc_cfg_ckpt, feat_mu_ckpt, feat_sigma_ckpt = load_mfcc_mlp_2cls(ckpt_mfcc, device)

    mfcc_cfg = {
        "sr": int(mfcc_cfg_ckpt.get("sr", getattr(cfg, "FS", 8192))),
        "n_fft": int(mfcc_cfg_ckpt.get("n_fft", 256)),
        "win_length": int(mfcc_cfg_ckpt.get("win_length", 256)),
        "hop_length": int(mfcc_cfg_ckpt.get("hop_length", 128)),
        "window": str(mfcc_cfg_ckpt.get("window", "hamming")),
        "n_mels": int(mfcc_cfg_ckpt.get("n_mels", 40)),
        "n_mfcc": int(mfcc_cfg_ckpt.get("n_mfcc", 20)),
        "fmin": float(mfcc_cfg_ckpt.get("fmin", 20.0)),
        "fmax": float(mfcc_cfg_ckpt.get("fmax", float(getattr(cfg, "FS", 8192)) / 2.0)),
        "center": bool(mfcc_cfg_ckpt.get("center", True)),
    }
    if mfcc_cfg["fmax"] <= 0:
        mfcc_cfg["fmax"] = float(mfcc_cfg["sr"]) / 2.0

    dl_tr, dl_va, dl_te = build_stft_loaders(args)
    z_dark_tr, y_tr_dark = logits_from_darknet(model_dark, dl_tr, device)
    z_dark_va, y_va_dark = logits_from_darknet(model_dark, dl_va, device)
    z_dark_te, y_te_dark = logits_from_darknet(model_dark, dl_te, device)

    raw = load_raw_splits(args.data_root_raw)
    y_tr = raw["y_train"].astype(np.int64)
    y_va = raw["y_val"].astype(np.int64)
    y_te = raw["y_test"].astype(np.int64)

    if not (np.array_equal(y_tr, y_tr_dark) and np.array_equal(y_va, y_va_dark) and np.array_equal(y_te, y_te_dark)):
        raise ValueError("Raw/MFCC 与 STFT/DarkNet 标签顺序不一致，无法逐样本融合。")

    print("Extracting MFCC features for fusion ...")
    f_tr = extract_mfcc_stats_split(raw["X_train"], mfcc_cfg, tag="train")
    f_va = extract_mfcc_stats_split(raw["X_val"], mfcc_cfg, tag="val")
    f_te = extract_mfcc_stats_split(raw["X_test"], mfcc_cfg, tag="test")

    if feat_mu_ckpt is not None and feat_sigma_ckpt is not None:
        mu = feat_mu_ckpt.astype(np.float32)
        sg = (feat_sigma_ckpt.astype(np.float32) + 1e-6)
    else:
        mu = f_tr.mean(axis=0, keepdims=True)
        sg = f_tr.std(axis=0, keepdims=True) + 1e-6
    f_tr = (f_tr - mu) / sg
    f_va = (f_va - mu) / sg
    f_te = (f_te - mu) / sg

    z_mfcc_tr = logits_from_mlp(model_mfcc, f_tr, args.batch_size, device)
    z_mfcc_va = logits_from_mlp(model_mfcc, f_va, args.batch_size, device)
    z_mfcc_te = logits_from_mlp(model_mfcc, f_te, args.batch_size, device)

    fusion = CWGFFusion2Cls(hid=int(args.fusion_hid), drop=float(args.fusion_drop)).to(device)

    zm_tr_t = torch.tensor(z_mfcc_tr, dtype=torch.float32, device=device)
    zd_tr_t = torch.tensor(z_dark_tr, dtype=torch.float32, device=device)
    zm_va_t = torch.tensor(z_mfcc_va, dtype=torch.float32, device=device)
    zd_va_t = torch.tensor(z_dark_va, dtype=torch.float32, device=device)
    zm_te_t = torch.tensor(z_mfcc_te, dtype=torch.float32, device=device)
    zd_te_t = torch.tensor(z_dark_te, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    fusion.fit_feature_norm_from_logits(zm_tr_t, zd_tr_t)

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    w_pos = (neg / max(pos, 1.0)) ** float(args.pos_pow)
    w_neg = 1.0
    print(f"[FusionTrain] train/val/test = {len(y_tr)}/{len(y_va)}/{len(y_te)}")
    print(f"[FusionTrain] w_pos={w_pos:.4f} (neg={neg:.0f}, pos={pos:.0f}, pos_pow={args.pos_pow})")

    opt = torch.optim.AdamW(fusion.parameters(), lr=float(args.fusion_lr), weight_decay=float(args.fusion_wd))

    best_val_f1 = -1.0
    best_val_acc = -1.0
    best_tau = 0.5
    best_state = None

    for ep in range(1, int(args.fusion_epochs) + 1):
        fusion.train()
        opt.zero_grad(set_to_none=True)
        p_tr, aux_tr = fusion(zm_tr_t, zd_tr_t)
        loss = bce_on_probs(p_tr, y_tr_t, w_pos=w_pos, w_neg=w_neg) - float(args.lambda_ent) * entropy_pos(p_tr)

        if "expert_weights" in aux_tr and float(args.lambda_center) > 0:
            loss = loss + float(args.lambda_center) * (aux_tr["expert_weights"] - 0.5).pow(2).mean()

        loss.backward()
        if float(args.grad_clip) > 0:
            nn.utils.clip_grad_norm_(fusion.parameters(), float(args.grad_clip))
        opt.step()

        if ep == 1 or ep % int(args.eval_every) == 0 or ep == int(args.fusion_epochs):
            fusion.eval()
            with torch.no_grad():
                p_va = fusion(zm_va_t, zd_va_t)[0].detach().cpu().numpy()
            tau, _ = find_best_thr(
                p_va,
                y_va,
                lo=float(args.thr_lo),
                hi=float(args.thr_hi),
                step=float(args.thr_step),
                objective=str(args.thr_objective),
            )
            m_va = bin_metrics_from_probs(p_va, y_va, tau)
            score = m_va["f1"] if str(args.thr_objective) == "f1" else m_va["acc"]
            best_score = best_val_f1 if str(args.thr_objective) == "f1" else best_val_acc
            if score > best_score:
                best_val_f1 = m_va["f1"]
                best_val_acc = m_va["acc"]
                best_tau = float(tau)
                best_state = {k: v.detach().cpu() for k, v in fusion.state_dict().items()}
            print(
                f"Epoch {ep:03d} | loss={loss.item():.4f} | val@tau={tau:.3f} "
                f"acc={m_va['acc']:.4f} f1={m_va['f1']:.4f} | best_tau={best_tau:.3f} best_f1={best_val_f1:.4f}"
            )

    if best_state is not None:
        fusion.load_state_dict(best_state, strict=True)

    fusion.eval()
    with torch.no_grad():
        p_te = fusion(zm_te_t, zd_te_t)[0].detach().cpu().numpy()
    m_te = bin_metrics_from_probs(p_te, y_te, best_tau)

    print(
        f"[TEST] thr={best_tau:.3f} | acc={m_te['acc']:.4f} f1={m_te['f1']:.4f} "
        f"P={m_te['precision']:.4f} R={m_te['recall']:.4f}"
    )
    print(f"[TEST] Confusion Matrix [[TN,FP],[FN,TP]]:\n{m_te['cm']}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if bool(args.save_ckpt):
        ckpt_path = os.path.join(args.save_dir, f"fuse_cwgf_mfcc_darknet_2cls_{stamp}.pt")
        torch.save(
            {
                "model": "CWGF_MFCC_DARKNET_2CLS",
                "state_dict": fusion.state_dict(),
                "best_thr": best_tau,
                "best_val_f1": best_val_f1,
                "mfcc_cfg": mfcc_cfg,
                "seed": int(args.seed),
            },
            ckpt_path,
        )
        print(f"[CKPT] {ckpt_path}")

    log_path = os.path.join(args.log_dir, f"metrics_fuse_cwgf_mfcc_darknet_2cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "val_best_f1",
                "val_best_acc",
                "val_best_thr",
                "test_acc",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_bal_acc",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "method": "cwgf",
                "val_best_f1": f"{best_val_f1:.6f}",
                "val_best_acc": f"{best_val_acc:.6f}",
                "val_best_thr": f"{best_tau:.6f}",
                "test_acc": f"{m_te['acc']:.6f}",
                "test_precision": f"{m_te['precision']:.6f}",
                "test_recall": f"{m_te['recall']:.6f}",
                "test_f1": f"{m_te['f1']:.6f}",
                "test_bal_acc": f"{m_te['bal_acc']:.6f}",
            }
        )
    print(f"[LOG] {log_path}")


def main():
    args = build_parser().parse_args()
    run_fusion(args)


if __name__ == "__main__":
    main()
