import argparse
import csv
import glob
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchvision.models import resnet18

import config as cfg
from Darknet19 import Darknet19
from NpyDataset import NpyDataset
from fusion_2cls_models import CWGFFusion2Cls, build_fusion_2cls


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


@torch.no_grad()
def logits_from_model(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if out.ndim != 2 or out.shape[1] != 2:
            raise ValueError(f"Expected [B,2] logits, got {tuple(out.shape)}")
        logits_all.append(out.detach().cpu().numpy())
        y_all.append(y.numpy().astype(np.int64))
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def prob_pos_from_logits_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / np.maximum(ez.sum(axis=1, keepdims=True), 1e-12)
    return np.clip(p[:, 1], 1e-9, 1.0 - 1e-9)


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

    scores = np.array([_score(float(t)) for t in taus])
    i = int(np.argmax(scores))
    tau1 = float(taus[i])

    lo2 = max(lo, tau1 - 5 * step)
    hi2 = min(hi, tau1 + 5 * step)
    step2 = step / 10.0
    n2 = int(np.floor((hi2 - lo2) / step2)) + 1
    taus2 = np.round(np.linspace(lo2, hi2, n2), 6)
    scores2 = np.array([_score(float(t)) for t in taus2])
    j = int(np.argmax(scores2))

    return float(taus2[j]), float(scores2[j])


def bce_on_probs(p: torch.Tensor, y: torch.Tensor, w_pos: float = 1.0, w_neg: float = 1.0, eps: float = 1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return (-w_pos * y * torch.log(p) - w_neg * (1 - y) * torch.log(1 - p)).mean()


def entropy_pos(p: torch.Tensor, eps: float = 1e-9):
    p = torch.clamp(p, eps, 1.0 - eps)
    return (-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).mean()


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def load_resnet_2cls(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))
    num_classes = int(ck.get("num_classes", 2))
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state, strict=False)
    model.to(device)
    freeze_model(model)
    return model


def load_darknet_2cls(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))
    num_classes = int(ck.get("num_classes", 2))
    model = Darknet19(num_classes=num_classes)
    model.load_state_dict(state, strict=False)
    model.to(device)
    freeze_model(model)
    return model


def build_loaders(args):
    root = args.data_root
    X_tr = os.path.join(root, f"X_train_stft_{args.img_size}.npy")
    X_va = os.path.join(root, f"X_val_stft_{args.img_size}.npy")
    X_te = os.path.join(root, f"X_test_stft_{args.img_size}.npy")
    y_tr = os.path.join(root, "y_train.npy")
    y_va = os.path.join(root, "y_val.npy")
    y_te = os.path.join(root, "y_test.npy")

    required = [X_tr, X_va, X_te, y_tr, y_va, y_te]
    miss = [p for p in required if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("Missing dataset files:\n" + "\n".join(miss))

    ds_tr = NpyDataset(X_tr, y_tr, normalize="imagenet", memmap=True)
    ds_va = NpyDataset(X_va, y_va, normalize="imagenet", memmap=True)
    ds_te = NpyDataset(X_te, y_te, normalize="imagenet", memmap=True)

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


def build_parser(default_method: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion_method", type=str, default=default_method, choices=["cwgf", "attention", "moe"])

    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt_resnet", type=str, default="")
    ap.add_argument("--ckpt_darknet", type=str, default="")

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
    ap.add_argument("--seed", type=int, default=getattr(cfg, "SEED", 2025))

    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--save_ckpt", type=int, default=1)
    ap.add_argument("--dry_run", type=int, default=0)
    return ap


def run_fusion(args) -> None:
    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.log_dir, exist_ok=True)
    if bool(args.save_ckpt):
        os.makedirs(args.save_dir, exist_ok=True)

    ckpt_res = resolve_latest_ckpt(args.ckpt_resnet, "stft_resnet18_2cls_best_*.pth", "resnet")
    ckpt_dark = resolve_latest_ckpt(args.ckpt_darknet, "darknet19_best_*.pth", "darknet")

    print("========== Fuse 2cls (frozen experts) ==========")
    print(f"method      : {args.fusion_method}")
    print(f"device      : {device}")
    print(f"data_root   : {args.data_root}")
    print(f"ckpt_resnet : {ckpt_res}")
    print(f"ckpt_darknet: {ckpt_dark}")
    print("===============================================")

    if bool(args.dry_run):
        return

    dl_tr, dl_va, dl_te = build_loaders(args)

    model_res = load_resnet_2cls(ckpt_res, device)
    model_dark = load_darknet_2cls(ckpt_dark, device)

    z_res_tr, y_tr = logits_from_model(model_res, dl_tr, device)
    z_res_va, y_va = logits_from_model(model_res, dl_va, device)
    z_res_te, y_te = logits_from_model(model_res, dl_te, device)

    z_dark_tr, y2_tr = logits_from_model(model_dark, dl_tr, device)
    z_dark_va, y2_va = logits_from_model(model_dark, dl_va, device)
    z_dark_te, y2_te = logits_from_model(model_dark, dl_te, device)

    if not (np.array_equal(y_tr, y2_tr) and np.array_equal(y_va, y2_va) and np.array_equal(y_te, y2_te)):
        raise ValueError("ResNet 与 DarkNet 的标签顺序不一致，无法逐样本融合。")

    fusion = build_fusion_2cls(args.fusion_method, hidden_dim=args.fusion_hid, dropout=args.fusion_drop).to(device)

    zr_tr_t = torch.tensor(z_res_tr, dtype=torch.float32, device=device)
    zd_tr_t = torch.tensor(z_dark_tr, dtype=torch.float32, device=device)
    zr_va_t = torch.tensor(z_res_va, dtype=torch.float32, device=device)
    zd_va_t = torch.tensor(z_dark_va, dtype=torch.float32, device=device)
    zr_te_t = torch.tensor(z_res_te, dtype=torch.float32, device=device)
    zd_te_t = torch.tensor(z_dark_te, dtype=torch.float32, device=device)

    if isinstance(fusion, CWGFFusion2Cls):
        fusion.fit_feature_norm_from_logits(zr_tr_t, zd_tr_t)

    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    pos = float((y_tr == 1).sum())
    neg = float((y_tr == 0).sum())
    w_pos = (neg / max(pos, 1.0)) ** float(args.pos_pow)
    w_neg = 1.0

    print(f"[FusionTrain] train/val/test = {len(y_tr)}/{len(y_va)}/{len(y_te)}")
    print(f"[FusionTrain] w_pos={w_pos:.4f} (neg={neg:.0f}, pos={pos:.0f}, pos_pow={args.pos_pow})")

    opt = torch.optim.AdamW(fusion.parameters(), lr=args.fusion_lr, weight_decay=args.fusion_wd)

    best_val_f1 = -1.0
    best_tau = 0.5
    best_state = None

    for ep in range(1, int(args.fusion_epochs) + 1):
        fusion.train()
        opt.zero_grad(set_to_none=True)
        p_tr, aux_tr = fusion(zr_tr_t, zd_tr_t)

        bce = bce_on_probs(p_tr, y_tr_t, w_pos=w_pos, w_neg=w_neg)
        loss = bce - float(args.lambda_ent) * entropy_pos(p_tr)

        if "expert_weights" in aux_tr and float(args.lambda_center) > 0:
            loss = loss + float(args.lambda_center) * (aux_tr["expert_weights"] - 0.5).pow(2).mean()

        loss.backward()
        if float(args.grad_clip) > 0:
            nn.utils.clip_grad_norm_(fusion.parameters(), float(args.grad_clip))
        opt.step()

        if ep == 1 or ep % int(args.eval_every) == 0 or ep == int(args.fusion_epochs):
            fusion.eval()
            with torch.no_grad():
                p_va, _ = fusion(zr_va_t, zd_va_t)
                p_va_np = p_va.detach().cpu().numpy()

            tau, _ = find_best_thr(
                p_va_np,
                y_va,
                lo=float(args.thr_lo),
                hi=float(args.thr_hi),
                step=float(args.thr_step),
                objective=str(args.thr_objective),
            )
            m_va = bin_metrics_from_probs(p_va_np, y_va, tau)

            print(
                f"Epoch {ep:03d} | loss={loss.item():.4f} | "
                f"val@{tau:.3f} f1={m_va['f1']:.4f} acc={m_va['acc']:.4f} | "
                f"best@{best_tau:.3f} f1={best_val_f1:.4f}"
            )

            if m_va["f1"] > best_val_f1:
                best_val_f1 = m_va["f1"]
                best_tau = tau
                best_state = {k: v.detach().cpu() for k, v in fusion.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in fusion.state_dict().items()}
    fusion.load_state_dict(best_state, strict=True)
    fusion.eval()

    with torch.no_grad():
        p_te, _ = fusion(zr_te_t, zd_te_t)
        p_te_np = p_te.detach().cpu().numpy()

    m_fuse = bin_metrics_from_probs(p_te_np, y_te, best_tau)

    p_res_te = prob_pos_from_logits_np(z_res_te)
    p_dark_te = prob_pos_from_logits_np(z_dark_te)
    m_res = bin_metrics_from_probs(p_res_te, y_te, 0.5)
    m_dark = bin_metrics_from_probs(p_dark_te, y_te, 0.5)

    print(
        f"\n[FUSE-{args.fusion_method.upper()}] TEST @tau={best_tau:.3f} | "
        f"acc={m_fuse['acc']:.4f} f1={m_fuse['f1']:.4f} "
        f"prec={m_fuse['precision']:.4f} rec={m_fuse['recall']:.4f}"
    )
    print("[FUSE] Confusion Matrix [[TN,FP],[FN,TP]]:\n", m_fuse["cm"])
    print(
        f"[BASE] STFT-ResNet18(2cls) @0.5 | acc={m_res['acc']:.4f} f1={m_res['f1']:.4f}"
    )
    print(
        f"[BASE] DarkNet19(2cls)      @0.5 | acc={m_dark['acc']:.4f} f1={m_dark['f1']:.4f}"
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    method = args.fusion_method.lower()

    ckpt_path = ""
    if bool(args.save_ckpt):
        ckpt_path = os.path.join(args.save_dir, f"fuse2_{method}_resnet_darknet_best_{stamp}.pt")
        torch.save(
            {
                "method": method,
                "state_dict": {k: v.cpu() for k, v in fusion.state_dict().items()},
                "best_tau": float(best_tau),
                "best_val_f1": float(best_val_f1),
                "ckpt_resnet": ckpt_res,
                "ckpt_darknet": ckpt_dark,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"[CKPT] saved: {ckpt_path}")

    log_path = os.path.join(args.log_dir, f"metrics_fuse2_{method}_resnet_darknet_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "tau", "fuse_acc", "fuse_f1", "fuse_prec", "fuse_rec", "fuse_bal_acc", "best_val_f1"])
        w.writerow(
            [
                method,
                f"{best_tau:.4f}",
                f"{m_fuse['acc']:.6f}",
                f"{m_fuse['f1']:.6f}",
                f"{m_fuse['precision']:.6f}",
                f"{m_fuse['recall']:.6f}",
                f"{m_fuse['bal_acc']:.6f}",
                f"{best_val_f1:.6f}",
            ]
        )
        w.writerow([])
        w.writerow(["expert", "tau", "acc", "f1", "precision", "recall", "bal_acc"])
        w.writerow(
            [
                "stft_resnet18_2cls",
                "0.5000",
                f"{m_res['acc']:.6f}",
                f"{m_res['f1']:.6f}",
                f"{m_res['precision']:.6f}",
                f"{m_res['recall']:.6f}",
                f"{m_res['bal_acc']:.6f}",
            ]
        )
        w.writerow(
            [
                "darknet19_2cls",
                "0.5000",
                f"{m_dark['acc']:.6f}",
                f"{m_dark['f1']:.6f}",
                f"{m_dark['precision']:.6f}",
                f"{m_dark['recall']:.6f}",
                f"{m_dark['bal_acc']:.6f}",
            ]
        )

    print(f"[LOG] saved: {log_path}")


def main_entry(default_method: str) -> None:
    parser = build_parser(default_method=default_method)
    args = parser.parse_args()
    args.fusion_method = default_method
    run_fusion(args)


if __name__ == "__main__":
    parser = build_parser(default_method="cwgf")
    args = parser.parse_args()
    run_fusion(args)
