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
from joblib import load as joblib_load
from sklearn.metrics import confusion_matrix, f1_score, recall_score

import config as cfg
from Gatenet import Gatenet


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


def safe_torch_load(path: str, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    ck = safe_torch_load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))
    in_dim, hidden, out_dim = _find_linear_dims_from_state_dict(state)
    num_classes = int(ck.get("num_classes", out_dim))

    model = MLP4(in_dim=in_dim, hidden=hidden, p_drop=0.0, num_classes=num_classes)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    mfcc_cfg = dict(ck.get("mfcc_cfg", {}))
    feat_mu = ck.get("feat_mu", None)
    feat_sigma = ck.get("feat_sigma", None)
    mu = np.asarray(feat_mu, dtype=np.float32).reshape(1, -1) if feat_mu is not None else None
    sg = np.asarray(feat_sigma, dtype=np.float32).reshape(1, -1) if feat_sigma is not None else None
    return model, mfcc_cfg, mu, sg


def load_mfcc_svm_4cls(ckpt_path: str):
    ck = joblib_load(ckpt_path)
    if not isinstance(ck, dict) or "model" not in ck:
        raise ValueError(f"Unexpected SVM checkpoint format: {ckpt_path}")
    model = ck["model"]
    mfcc_cfg = dict(ck.get("mfcc_cfg", {}))
    feat_mu = ck.get("feat_mu", None)
    feat_sigma = ck.get("feat_sigma", None)
    mu = np.asarray(feat_mu, dtype=np.float32).reshape(1, -1) if feat_mu is not None else None
    sg = np.asarray(feat_sigma, dtype=np.float32).reshape(1, -1) if feat_sigma is not None else None
    return model, mfcc_cfg, mu, sg


def logits_from_mlp(model: nn.Module, feats: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    n = feats.shape[0]
    for i in range(0, n, batch_size):
        x = torch.from_numpy(feats[i : i + batch_size]).to(device)
        logits = model(x)
        outs.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


def logits_from_svm(model, feats: np.ndarray) -> np.ndarray:
    if not hasattr(model, "decision_function"):
        raise ValueError("SVM model has no decision_function")
    z = model.decision_function(feats)
    z = np.asarray(z, dtype=np.float32)
    if z.ndim == 1:
        z = np.stack([-z, z], axis=1)
    if z.ndim != 2:
        raise ValueError(f"Unexpected SVM decision_function shape: {z.shape}")
    # For 4cls fusion we expect [N,4] logits-like scores.
    if z.shape[1] != 4:
        raise ValueError(
            f"SVM decision_function output must be [N,4] for this script, got {z.shape}. "
            "Please ensure SVC(decision_function_shape='ovr') for 4 classes."
        )
    return z.astype(np.float32)


def probs_from_logits_np(z: np.ndarray) -> np.ndarray:
    zmax = z.max(axis=1, keepdims=True)
    ez = np.exp(z - zmax)
    return ez / np.maximum(ez.sum(axis=1, keepdims=True), 1e-12)


def entropy_categorical_np(p: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def build_per_class_features(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    if p_a.shape != p_b.shape or p_a.ndim != 2:
        raise ValueError(f"Invalid prob shape: {p_a.shape} vs {p_b.shape}")
    n, c = p_a.shape
    h_a = entropy_categorical_np(p_a)
    h_b = entropy_categorical_np(p_b)
    h_a_c = np.repeat(h_a[:, None], c, axis=1)
    h_b_c = np.repeat(h_b[:, None], c, axis=1)
    feats = np.stack([p_a, p_b, np.abs(p_a - p_b), h_a_c, h_b_c], axis=2)
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


def fuse_logits_per_class(z_a: torch.Tensor, z_b: torch.Tensor, g4: torch.Tensor) -> torch.Tensor:
    la = F.log_softmax(z_a, dim=1)
    lb = F.log_softmax(z_b, dim=1)
    return g4 * la + (1.0 - g4) * lb


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
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--ckpt_mfcc_svm4", type=str, default="")
    ap.add_argument("--ckpt_mfcc_mlp4", type=str, default="")
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

    ckpt_svm = resolve_latest_ckpt(args.ckpt_mfcc_svm4, "svm_mfcc_4cls_best_*.joblib", "mfcc_svm4")
    ckpt_mlp = resolve_latest_ckpt(args.ckpt_mfcc_mlp4, "mlp_mfcc_4cls_best_*.pt", "mfcc_mlp4")

    print("========== Fuse MFCC-SVM + MFCC-MLP (4cls, CWGF-like) ==========")
    print(f"device        : {device}")
    print(f"data_root_raw : {args.data_root_raw}")
    print(f"ckpt_mfcc_svm4: {ckpt_svm}")
    print(f"ckpt_mfcc_mlp4: {ckpt_mlp}")
    print("=================================================================")

    svm_model, svm_cfg_ckpt, svm_mu_ckpt, svm_sg_ckpt = load_mfcc_svm_4cls(ckpt_svm)
    mlp_model, mlp_cfg_ckpt, mlp_mu_ckpt, mlp_sg_ckpt = load_mfcc_mlp_4cls(ckpt_mlp, device)

    cfg_svm = {
        "sr": int(svm_cfg_ckpt.get("sr", getattr(cfg, "FS", 8192))),
        "n_fft": int(svm_cfg_ckpt.get("n_fft", 256)),
        "win_length": int(svm_cfg_ckpt.get("win_length", 256)),
        "hop_length": int(svm_cfg_ckpt.get("hop_length", 128)),
        "window": str(svm_cfg_ckpt.get("window", "hamming")),
        "n_mels": int(svm_cfg_ckpt.get("n_mels", 40)),
        "n_mfcc": int(svm_cfg_ckpt.get("n_mfcc", 20)),
        "fmin": float(svm_cfg_ckpt.get("fmin", 20.0)),
        "fmax": float(svm_cfg_ckpt.get("fmax", float(getattr(cfg, "FS", 8192)) / 2.0)),
        "center": bool(svm_cfg_ckpt.get("center", True)),
    }
    cfg_mlp = {
        "sr": int(mlp_cfg_ckpt.get("sr", getattr(cfg, "FS", 8192))),
        "n_fft": int(mlp_cfg_ckpt.get("n_fft", 256)),
        "win_length": int(mlp_cfg_ckpt.get("win_length", 256)),
        "hop_length": int(mlp_cfg_ckpt.get("hop_length", 128)),
        "window": str(mlp_cfg_ckpt.get("window", "hamming")),
        "n_mels": int(mlp_cfg_ckpt.get("n_mels", 40)),
        "n_mfcc": int(mlp_cfg_ckpt.get("n_mfcc", 20)),
        "fmin": float(mlp_cfg_ckpt.get("fmin", 20.0)),
        "fmax": float(mlp_cfg_ckpt.get("fmax", float(getattr(cfg, "FS", 8192)) / 2.0)),
        "center": bool(mlp_cfg_ckpt.get("center", True)),
    }
    if cfg_svm["fmax"] <= 0:
        cfg_svm["fmax"] = float(cfg_svm["sr"]) / 2.0
    if cfg_mlp["fmax"] <= 0:
        cfg_mlp["fmax"] = float(cfg_mlp["sr"]) / 2.0

    raw = load_raw_splits_4cls(args.data_root_raw)
    y4_tr = raw["y4_train"]
    y4_va = raw["y4_val"]
    y4_te = raw["y4_test"]

    print("Extracting MFCC features (SVM config) ...")
    f_svm_tr = extract_mfcc_stats_split(raw["X_train"], cfg_svm, tag="train-svm")
    f_svm_va = extract_mfcc_stats_split(raw["X_val"], cfg_svm, tag="val-svm")
    f_svm_te = extract_mfcc_stats_split(raw["X_test"], cfg_svm, tag="test-svm")

    print("Extracting MFCC features (MLP config) ...")
    f_mlp_tr = extract_mfcc_stats_split(raw["X_train"], cfg_mlp, tag="train-mlp")
    f_mlp_va = extract_mfcc_stats_split(raw["X_val"], cfg_mlp, tag="val-mlp")
    f_mlp_te = extract_mfcc_stats_split(raw["X_test"], cfg_mlp, tag="test-mlp")

    if svm_mu_ckpt is not None and svm_sg_ckpt is not None:
        mu_svm, sg_svm = svm_mu_ckpt.astype(np.float32), svm_sg_ckpt.astype(np.float32) + 1e-6
    else:
        mu_svm = f_svm_tr.mean(axis=0, keepdims=True)
        sg_svm = f_svm_tr.std(axis=0, keepdims=True) + 1e-6
    f_svm_tr = (f_svm_tr - mu_svm) / sg_svm
    f_svm_va = (f_svm_va - mu_svm) / sg_svm
    f_svm_te = (f_svm_te - mu_svm) / sg_svm

    if mlp_mu_ckpt is not None and mlp_sg_ckpt is not None:
        mu_mlp, sg_mlp = mlp_mu_ckpt.astype(np.float32), mlp_sg_ckpt.astype(np.float32) + 1e-6
    else:
        mu_mlp = f_mlp_tr.mean(axis=0, keepdims=True)
        sg_mlp = f_mlp_tr.std(axis=0, keepdims=True) + 1e-6
    f_mlp_tr = (f_mlp_tr - mu_mlp) / sg_mlp
    f_mlp_va = (f_mlp_va - mu_mlp) / sg_mlp
    f_mlp_te = (f_mlp_te - mu_mlp) / sg_mlp

    z_svm_tr = logits_from_svm(svm_model, f_svm_tr)
    z_svm_va = logits_from_svm(svm_model, f_svm_va)
    z_svm_te = logits_from_svm(svm_model, f_svm_te)

    z_mlp_tr = logits_from_mlp(mlp_model, f_mlp_tr, args.batch_size, device)
    z_mlp_va = logits_from_mlp(mlp_model, f_mlp_va, args.batch_size, device)
    z_mlp_te = logits_from_mlp(mlp_model, f_mlp_te, args.batch_size, device)

    p_svm_tr = probs_from_logits_np(z_svm_tr)
    p_svm_va = probs_from_logits_np(z_svm_va)
    p_svm_te = probs_from_logits_np(z_svm_te)
    p_mlp_tr = probs_from_logits_np(z_mlp_tr)
    p_mlp_va = probs_from_logits_np(z_mlp_va)
    p_mlp_te = probs_from_logits_np(z_mlp_te)

    feat_tr = build_per_class_features(p_svm_tr, p_mlp_tr)
    feat_va = build_per_class_features(p_svm_va, p_mlp_va)
    feat_te = build_per_class_features(p_svm_te, p_mlp_te)

    feat_tr_n, mu_phi, sg_phi = norm_feats(feat_tr, feat_tr)
    feat_va_n = ((feat_va - mu_phi) / sg_phi).astype(np.float32)
    feat_te_n = ((feat_te - mu_phi) / sg_phi).astype(np.float32)

    gate = Gatenet(in_dim=5, hid=int(args.gate_hid), drop=float(args.gate_drop)).to(device)
    init_gate_half(gate)
    opt = torch.optim.AdamW(gate.parameters(), lr=float(args.gate_lr), weight_decay=float(args.gate_wd))
    ce = nn.CrossEntropyLoss()

    zs_tr_t = torch.tensor(z_svm_tr, dtype=torch.float32, device=device)
    zm_tr_t = torch.tensor(z_mlp_tr, dtype=torch.float32, device=device)
    zs_va_t = torch.tensor(z_svm_va, dtype=torch.float32, device=device)
    zm_va_t = torch.tensor(z_mlp_va, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y4_tr, dtype=torch.long, device=device)
    feat_tr_t = torch.tensor(feat_tr_n, dtype=torch.float32, device=device)
    feat_va_t = torch.tensor(feat_va_n, dtype=torch.float32, device=device)

    best_val_macro = -1.0
    best_state = None

    for ep in range(1, int(args.gate_epochs) + 1):
        gate.train()
        opt.zero_grad(set_to_none=True)
        g_tr = gate(feat_tr_t.reshape(-1, feat_tr_t.shape[-1])).reshape(-1, 4)
        zf_tr = fuse_logits_per_class(zs_tr_t, zm_tr_t, g_tr)
        loss = ce(zf_tr, y_tr_t) + float(args.gate_center) * (g_tr - 0.5).pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()

        if ep == 1 or ep % int(args.eval_every) == 0 or ep == int(args.gate_epochs):
            gate.eval()
            with torch.no_grad():
                g_va = gate(feat_va_t.reshape(-1, feat_va_t.shape[-1])).reshape(-1, 4)
                zf_va = fuse_logits_per_class(zs_va_t, zm_va_t, g_va)
                p_va = F.softmax(zf_va, dim=1).detach().cpu().numpy()
            m_va = metrics_4cls_and_parent(p_va, y4_va)
            if m_va["macro_f1"] > best_val_macro:
                best_val_macro = m_va["macro_f1"]
                best_state = {k: v.detach().cpu() for k, v in gate.state_dict().items()}
            print(
                f"Epoch {ep:03d} | loss={loss.item():.4f} | "
                f"val_macro_f1={m_va['macro_f1']:.4f} | best_macro_f1={best_val_macro:.4f}"
            )

    if best_state is not None:
        gate.load_state_dict(best_state, strict=True)
    gate.eval()

    with torch.no_grad():
        feat_va_t = torch.tensor(feat_va_n, dtype=torch.float32, device=device)
        feat_te_t = torch.tensor(feat_te_n, dtype=torch.float32, device=device)
        zs_va_t = torch.tensor(z_svm_va, dtype=torch.float32, device=device)
        zm_va_t = torch.tensor(z_mlp_va, dtype=torch.float32, device=device)
        zs_te_t = torch.tensor(z_svm_te, dtype=torch.float32, device=device)
        zm_te_t = torch.tensor(z_mlp_te, dtype=torch.float32, device=device)
        g_va = gate(feat_va_t.reshape(-1, feat_va_t.shape[-1])).reshape(-1, 4)
        g_te = gate(feat_te_t.reshape(-1, feat_te_t.shape[-1])).reshape(-1, 4)
        zf_va = fuse_logits_per_class(zs_va_t, zm_va_t, g_va)
        zf_te = fuse_logits_per_class(zs_te_t, zm_te_t, g_te)
        p_fuse_va = F.softmax(zf_va, dim=1).detach().cpu().numpy()
        p_fuse_te = F.softmax(zf_te, dim=1).detach().cpu().numpy()

    m_fuse_va = metrics_4cls_and_parent(p_fuse_va, y4_va)
    m_fuse_te = metrics_4cls_and_parent(p_fuse_te, y4_te)
    m_svm_va = metrics_4cls_and_parent(p_svm_va, y4_va)
    m_mlp_va = metrics_4cls_and_parent(p_mlp_va, y4_va)
    m_svm_te = metrics_4cls_and_parent(p_svm_te, y4_te)
    m_mlp_te = metrics_4cls_and_parent(p_mlp_te, y4_te)

    best_single_val_f1 = max(m_svm_va["macro_f1"], m_mlp_va["macro_f1"])
    degeneracy = "degenerate" if m_fuse_va["macro_f1"] < best_single_val_f1 else "not degenerate"

    print(
        f"[VAL] SVM f1={m_svm_va['macro_f1']:.4f} | MLP f1={m_mlp_va['macro_f1']:.4f} | "
        f"CWGF f1={m_fuse_va['macro_f1']:.4f} => {degeneracy}"
    )
    print(
        f"[TEST] SVM f1={m_svm_te['macro_f1']:.4f} | MLP f1={m_mlp_te['macro_f1']:.4f} | "
        f"CWGF f1={m_fuse_te['macro_f1']:.4f}"
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if bool(args.save_ckpt):
        ckpt_path = os.path.join(args.save_dir, f"fuse_cwgf_mfccsvm_mfccmlp_4cls_{stamp}.pt")
        torch.save(
            {
                "model": "CWGF_MFCCSVM_MFCCMLP_4CLS",
                "state_dict": gate.state_dict(),
                "phi_mu": mu_phi,
                "phi_std": sg_phi,
                "best_val_macro_f1": best_val_macro,
                "seed": int(args.seed),
            },
            ckpt_path,
        )
        print(f"[CKPT] {ckpt_path}")

    log_path = os.path.join(args.log_dir, f"metrics_fuse_cwgf_mfccsvm_mfccmlp_4cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "val_svm_macro_f1",
                "val_mlp_macro_f1",
                "val_fuse_macro_f1",
                "degeneracy",
                "test_svm_macro_f1",
                "test_mlp_macro_f1",
                "test_fuse_macro_f1",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "method": "cwgf_mfccsvm_mfccmlp_4cls",
                "val_svm_macro_f1": f"{m_svm_va['macro_f1']:.6f}",
                "val_mlp_macro_f1": f"{m_mlp_va['macro_f1']:.6f}",
                "val_fuse_macro_f1": f"{m_fuse_va['macro_f1']:.6f}",
                "degeneracy": degeneracy,
                "test_svm_macro_f1": f"{m_svm_te['macro_f1']:.6f}",
                "test_mlp_macro_f1": f"{m_mlp_te['macro_f1']:.6f}",
                "test_fuse_macro_f1": f"{m_fuse_te['macro_f1']:.6f}",
            }
        )
    print(f"[LOG] {log_path}")


def main():
    args = build_parser().parse_args()
    run_fusion(args)


if __name__ == "__main__":
    main()

