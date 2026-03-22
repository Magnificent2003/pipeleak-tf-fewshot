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
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, recall_score

import config as cfg
from Darknet19 import Darknet19
from Gatenet import Gatenet
from NpyDataset import NpyDataset


class MLP4(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, p_drop: float = 0.2, num_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x):
        return self.net(x)


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


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def _find_linear_dims_from_state_dict(state: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    w1 = state.get("net.0.weight", None)
    w2 = state.get("net.3.weight", None)
    if w1 is None or w2 is None:
        raise KeyError("Cannot infer MLP4 dims from checkpoint state_dict (need net.0.weight and net.3.weight).")
    hidden = int(w1.shape[0])
    in_dim = int(w1.shape[1])
    out_dim = int(w2.shape[0])
    if int(w2.shape[1]) != hidden:
        raise ValueError(f"Unexpected MLP4 shape: net.0={tuple(w1.shape)}, net.3={tuple(w2.shape)}")
    return in_dim, hidden, out_dim


def load_mfcc_mlp_4cls(
    ckpt_path: str, device: torch.device
) -> Tuple[nn.Module, Dict[str, float], np.ndarray, np.ndarray]:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))

    in_dim, hidden, out_dim = _find_linear_dims_from_state_dict(state)
    num_classes = int(ck.get("num_classes", out_dim))

    model = MLP4(in_dim=in_dim, hidden=hidden, p_drop=0.0, num_classes=num_classes)
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


def load_darknet_4cls(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))
    num_classes = int(ck.get("num_classes", 4))
    model = Darknet19(num_classes=num_classes)
    model.load_state_dict(state, strict=False)
    model.to(device)
    freeze_model(model)
    return model


@torch.no_grad()
def logits_from_darknet(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if out.ndim != 2:
            raise ValueError(f"Expected [B,C] logits from DarkNet, got {tuple(out.shape)}")
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
        logits = model(x)
        outs.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def probs_from_logits_np(z: np.ndarray) -> np.ndarray:
    zmax = z.max(axis=1, keepdims=True)
    ez = np.exp(z - zmax)
    return ez / np.maximum(ez.sum(axis=1, keepdims=True), 1e-12)


def entropy_categorical_np(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def build_per_class_features(p_dark: np.ndarray, p_mfcc: np.ndarray) -> np.ndarray:
    if p_dark.shape != p_mfcc.shape or p_dark.ndim != 2:
        raise ValueError(f"Invalid prob shape: {p_dark.shape} vs {p_mfcc.shape}")
    n, c = p_dark.shape
    h_dark = entropy_categorical_np(p_dark)
    h_mfcc = entropy_categorical_np(p_mfcc)
    h_dark_c = np.repeat(h_dark[:, None], c, axis=1)
    h_mfcc_c = np.repeat(h_mfcc[:, None], c, axis=1)
    feats = np.stack([p_dark, p_mfcc, np.abs(p_dark - p_mfcc), h_dark_c, h_mfcc_c], axis=2)
    return feats.astype(np.float32)


def norm_feats(feat_tr: np.ndarray, feat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = feat_tr.reshape(-1, feat_tr.shape[-1]).mean(axis=0, keepdims=True)
    sg = feat_tr.reshape(-1, feat_tr.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    return ((feat - mu) / sg).astype(np.float32), mu.astype(np.float32), sg.astype(np.float32)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def macro_rec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))


def parent_metrics_from_cm4(cm4: np.ndarray):
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


def metrics_4cls_and_parent(p: np.ndarray, y4: np.ndarray):
    yhat = p.argmax(axis=1).astype(np.int64)
    cm4 = confusion_matrix(y4, yhat, labels=[0, 1, 2, 3])
    cm_parent, parent_f1, parent_rec = parent_metrics_from_cm4(cm4)
    return {
        "macro_f1": macro_f1(y4, yhat),
        "macro_rec": macro_rec(y4, yhat),
        "parent_f1": parent_f1,
        "parent_rec": parent_rec,
        "cm4": cm4,
        "cm_parent": cm_parent,
    }


def init_gate_half(gate: nn.Module) -> None:
    linears = [m for m in gate.modules() if isinstance(m, nn.Linear)]
    if not linears:
        raise RuntimeError("No nn.Linear found in gate.")
    for m in linears[:-1]:
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    last = linears[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)


def fuse_logits_per_class(z_dark: torch.Tensor, z_mfcc: torch.Tensor, g4: torch.Tensor) -> torch.Tensor:
    ld = F.log_softmax(z_dark, dim=1)
    lm = F.log_softmax(z_mfcc, dim=1)
    return g4 * ld + (1.0 - g4) * lm


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
    y4_train = os.path.join(root, "y4_train.npy")
    y4_val = os.path.join(root, "y4_val.npy")
    y4_test = os.path.join(root, "y4_test.npy")

    required = [x_train, x_val, x_test, y4_train, y4_val, y4_test]
    miss = [p for p in required if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("Missing STFT dataset files:\n" + "\n".join(miss))

    ds_tr = NpyDataset(x_train, y4_train, normalize="imagenet", memmap=True)
    ds_va = NpyDataset(x_val, y4_val, normalize="imagenet", memmap=True)
    ds_te = NpyDataset(x_test, y4_test, normalize="imagenet", memmap=True)

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


def load_raw_splits_4cls(data_root_raw: str):
    paths = {
        "X_train": os.path.join(data_root_raw, "X_train.npy"),
        "X_val": os.path.join(data_root_raw, "X_val.npy"),
        "X_test": os.path.join(data_root_raw, "X_test.npy"),
        "y4_train": os.path.join(data_root_raw, "y4_train.npy"),
        "y4_val": os.path.join(data_root_raw, "y4_val.npy"),
        "y4_test": os.path.join(data_root_raw, "y4_test.npy"),
    }
    miss = [p for p in paths.values() if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("Missing raw 4cls dataset files:\n" + "\n".join(miss))

    out = {k: np.load(v) for k, v in paths.items()}
    out["y4_train"] = out["y4_train"].astype(np.int64)
    out["y4_val"] = out["y4_val"].astype(np.int64)
    out["y4_test"] = out["y4_test"].astype(np.int64)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_raw", type=str, default=cfg.DATASET)
    ap.add_argument("--data_root_stft", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt_mfcc4", type=str, default="")
    ap.add_argument("--ckpt_darknet4", type=str, default="")

    ap.add_argument("--gate_hid", type=int, default=16)
    ap.add_argument("--gate_drop", type=float, default=0.1)
    ap.add_argument("--gate_epochs", type=int, default=200)
    ap.add_argument("--gate_lr", type=float, default=1e-4)
    ap.add_argument("--gate_wd", type=float, default=1e-3)
    ap.add_argument("--gate_center", type=float, default=1e-3)
    ap.add_argument("--eval_every", type=int, default=10)
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

    ckpt_mfcc = resolve_latest_ckpt(args.ckpt_mfcc4, "mlp_mfcc_4cls_best_*.pt", "mfcc4")
    ckpt_dark = resolve_latest_ckpt(args.ckpt_darknet4, "darknet19_4cls_best_*.pth", "darknet4")

    print("========== Fuse MFCC+DarkNet (4cls, CWGF-like) ==========")
    print(f"device       : {device}")
    print(f"data_root_raw: {args.data_root_raw}")
    print(f"data_root_stft: {args.data_root_stft}")
    print(f"ckpt_mfcc4   : {ckpt_mfcc}")
    print(f"ckpt_darknet4: {ckpt_dark}")
    print("=========================================================")

    model_dark = load_darknet_4cls(ckpt_dark, device)
    model_mfcc, mfcc_cfg_ckpt, feat_mu_ckpt, feat_sigma_ckpt = load_mfcc_mlp_4cls(ckpt_mfcc, device)

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
    z_dark_tr, y4_tr_dark = logits_from_darknet(model_dark, dl_tr, device)
    z_dark_va, y4_va_dark = logits_from_darknet(model_dark, dl_va, device)
    z_dark_te, y4_te_dark = logits_from_darknet(model_dark, dl_te, device)

    raw = load_raw_splits_4cls(args.data_root_raw)
    y4_tr = raw["y4_train"]
    y4_va = raw["y4_val"]
    y4_te = raw["y4_test"]
    if not (
        np.array_equal(y4_tr, y4_tr_dark)
        and np.array_equal(y4_va, y4_va_dark)
        and np.array_equal(y4_te, y4_te_dark)
    ):
        raise ValueError("Raw/MFCC 与 STFT/DarkNet 的 y4 顺序不一致，无法逐样本融合。")

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

    p_dark_tr = probs_from_logits_np(z_dark_tr)
    p_dark_va = probs_from_logits_np(z_dark_va)
    p_dark_te = probs_from_logits_np(z_dark_te)
    p_mfcc_tr = probs_from_logits_np(z_mfcc_tr)
    p_mfcc_va = probs_from_logits_np(z_mfcc_va)
    p_mfcc_te = probs_from_logits_np(z_mfcc_te)

    feat_tr = build_per_class_features(p_dark_tr, p_mfcc_tr)
    feat_va = build_per_class_features(p_dark_va, p_mfcc_va)
    feat_te = build_per_class_features(p_dark_te, p_mfcc_te)

    feat_tr_n, mu_phi, sg_phi = norm_feats(feat_tr, feat_tr)
    feat_va_n = ((feat_va - mu_phi) / sg_phi).astype(np.float32)
    feat_te_n = ((feat_te - mu_phi) / sg_phi).astype(np.float32)

    gate = Gatenet(in_dim=5, hid=int(args.gate_hid), drop=float(args.gate_drop)).to(device)
    init_gate_half(gate)

    opt = torch.optim.AdamW(gate.parameters(), lr=float(args.gate_lr), weight_decay=float(args.gate_wd))
    ce = nn.CrossEntropyLoss()

    zd_tr_t = torch.tensor(z_dark_tr, dtype=torch.float32, device=device)
    zm_tr_t = torch.tensor(z_mfcc_tr, dtype=torch.float32, device=device)
    zd_va_t = torch.tensor(z_dark_va, dtype=torch.float32, device=device)
    zm_va_t = torch.tensor(z_mfcc_va, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y4_tr, dtype=torch.long, device=device)
    feat_tr_t = torch.tensor(feat_tr_n, dtype=torch.float32, device=device)
    feat_va_t = torch.tensor(feat_va_n, dtype=torch.float32, device=device)

    best_val_macro = -1.0
    best_val_parent = -1.0
    best_state = None

    for ep in range(1, int(args.gate_epochs) + 1):
        gate.train()
        opt.zero_grad(set_to_none=True)
        g_tr = gate(feat_tr_t.reshape(-1, feat_tr_t.shape[-1])).reshape(-1, 4)
        zf_tr = fuse_logits_per_class(zd_tr_t, zm_tr_t, g_tr)
        loss = ce(zf_tr, y_tr_t) + float(args.gate_center) * (g_tr - 0.5).pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()

        if ep == 1 or ep % int(args.eval_every) == 0 or ep == int(args.gate_epochs):
            gate.eval()
            with torch.no_grad():
                g_va = gate(feat_va_t.reshape(-1, feat_va_t.shape[-1])).reshape(-1, 4)
                zf_va = fuse_logits_per_class(zd_va_t, zm_va_t, g_va)
                p_va = F.softmax(zf_va, dim=1).detach().cpu().numpy()
            m_va = metrics_4cls_and_parent(p_va, y4_va)
            if m_va["macro_f1"] > best_val_macro:
                best_val_macro = m_va["macro_f1"]
                best_val_parent = m_va["parent_f1"]
                best_state = {k: v.detach().cpu() for k, v in gate.state_dict().items()}
            print(
                f"Epoch {ep:03d} | loss={loss.item():.4f} | val_macro_f1={m_va['macro_f1']:.4f} "
                f"| val_parent_f1={m_va['parent_f1']:.4f} | best_macro_f1={best_val_macro:.4f}"
            )

    if best_state is not None:
        gate.load_state_dict(best_state, strict=True)
    gate.eval()

    with torch.no_grad():
        feat_te_t = torch.tensor(feat_te_n, dtype=torch.float32, device=device)
        zd_te_t = torch.tensor(z_dark_te, dtype=torch.float32, device=device)
        zm_te_t = torch.tensor(z_mfcc_te, dtype=torch.float32, device=device)
        g_te = gate(feat_te_t.reshape(-1, feat_te_t.shape[-1])).reshape(-1, 4)
        zf_te = fuse_logits_per_class(zd_te_t, zm_te_t, g_te)
        p_fuse_te = F.softmax(zf_te, dim=1).detach().cpu().numpy()

    m_fuse = metrics_4cls_and_parent(p_fuse_te, y4_te)
    m_dark = metrics_4cls_and_parent(p_dark_te, y4_te)
    m_mfcc = metrics_4cls_and_parent(p_mfcc_te, y4_te)

    print(
        "[TEST][FUSE] macro_f1={:.4f} macro_rec={:.4f} parent_f1={:.4f} parent_rec={:.4f}".format(
            m_fuse["macro_f1"], m_fuse["macro_rec"], m_fuse["parent_f1"], m_fuse["parent_rec"]
        )
    )
    print("[TEST][DarkNet] macro_f1={:.4f} parent_f1={:.4f}".format(m_dark["macro_f1"], m_dark["parent_f1"]))
    print("[TEST][MFCC-MLP] macro_f1={:.4f} parent_f1={:.4f}".format(m_mfcc["macro_f1"], m_mfcc["parent_f1"]))
    print("[TEST][FUSE] cm4:\n", m_fuse["cm4"])

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if bool(args.save_ckpt):
        ckpt_path = os.path.join(args.save_dir, f"fuse_cwgf_mfcc_darknet_4cls_{stamp}.pt")
        torch.save(
            {
                "model": "CWGF_MFCC_DARKNET_4CLS",
                "state_dict": gate.state_dict(),
                "phi_mu": mu_phi,
                "phi_std": sg_phi,
                "best_val_macro_f1": best_val_macro,
                "seed": int(args.seed),
            },
            ckpt_path,
        )
        print(f"[CKPT] {ckpt_path}")

    log_path = os.path.join(args.log_dir, f"metrics_fuse_cwgf_mfcc_darknet_4cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "val_best_macro_f1",
                "val_best_parent_f1",
                "test_macro_f1",
                "test_macro_recall",
                "test_parent_f1",
                "test_parent_recall",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "method": "cwgf",
                "val_best_macro_f1": f"{best_val_macro:.6f}",
                "val_best_parent_f1": f"{best_val_parent:.6f}",
                "test_macro_f1": f"{m_fuse['macro_f1']:.6f}",
                "test_macro_recall": f"{m_fuse['macro_rec']:.6f}",
                "test_parent_f1": f"{m_fuse['parent_f1']:.6f}",
                "test_parent_recall": f"{m_fuse['parent_rec']:.6f}",
            }
        )
    print(f"[LOG] {log_path}")


def main():
    args = build_parser().parse_args()
    run_fusion(args)


if __name__ == "__main__":
    main()
