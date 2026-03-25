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
from fusion_2cls_models import CWGFFusion2Cls


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
            f"Please pass checkpoint path explicitly."
        )
    cands.sort(key=lambda p: os.path.getmtime(p))
    return cands[-1]


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


@torch.no_grad()
def logits_from_model(model: nn.Module, loader, device: torch.device, expected_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if out.ndim != 2 or out.shape[1] != expected_classes:
            raise ValueError(f"Expected [B,{expected_classes}] logits, got {tuple(out.shape)}")
        logits_all.append(out.detach().cpu().numpy())
        y_all.append(y.numpy().astype(np.int64))
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


def prob_pos_from_logits_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / np.maximum(ez.sum(axis=1, keepdims=True), 1e-12)
    return np.clip(p[:, 1], 1e-9, 1.0 - 1e-9)


def parent_prob_candidates_from_dark4_logits(logits4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    z = logits4 - logits4.max(axis=1, keepdims=True)
    ez = np.exp(z)
    p = ez / np.maximum(ez.sum(axis=1, keepdims=True), 1e-12)
    p_leak = np.clip(p[:, 0] + p[:, 1], 1e-9, 1.0 - 1e-9)      # parent class {0,1}
    p_no_leak = np.clip(p[:, 2] + p[:, 3], 1e-9, 1.0 - 1e-9)   # parent class {2,3}
    return p_leak, p_no_leak


def prob_to_2cls_logits_np(p_pos: np.ndarray) -> np.ndarray:
    p = np.clip(p_pos.astype(np.float64), 1e-9, 1.0 - 1e-9)
    logit = np.log(p) - np.log(1.0 - p)
    z = np.stack([np.zeros_like(logit), logit], axis=1)
    return z.astype(np.float32)


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


def entropy_pos(p: torch.Tensor, eps: float = 1e-6):
    p = torch.clamp(p, eps, 1.0 - eps)
    return (-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).mean()


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


def load_darknet_4cls(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))
    num_classes = int(ck.get("num_classes", 4))
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
        ds_tr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    dl_te = torch.utils.data.DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    return dl_tr, dl_va, dl_te


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt_resnet2", type=str, default="")
    ap.add_argument("--ckpt_darknet4", type=str, default="")
    ap.add_argument("--dark_parent_positive", type=str, default="auto", choices=["auto", "leak", "no_leak"])

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

    ckpt_res = resolve_latest_ckpt(args.ckpt_resnet2, "stft_resnet18_2cls_best_*.pth", "resnet2")
    ckpt_dark4 = resolve_latest_ckpt(args.ckpt_darknet4, "darknet19_4cls_best_*.pth", "darknet4")

    print("========== Fuse CWGF (DarkNet4-parent + ResNet2) ==========")
    print(f"device       : {device}")
    print(f"data_root    : {args.data_root}")
    print(f"ckpt_resnet2 : {ckpt_res}")
    print(f"ckpt_darknet4: {ckpt_dark4}")
    print("===========================================================")

    if bool(args.dry_run):
        return

    dl_tr, dl_va, dl_te = build_loaders(args)
    model_res = load_resnet_2cls(ckpt_res, device)
    model_dark4 = load_darknet_4cls(ckpt_dark4, device)

    z_res_tr, y_tr = logits_from_model(model_res, dl_tr, device, expected_classes=2)
    z_res_va, y_va = logits_from_model(model_res, dl_va, device, expected_classes=2)
    z_res_te, y_te = logits_from_model(model_res, dl_te, device, expected_classes=2)

    z_dark4_tr, y2_tr = logits_from_model(model_dark4, dl_tr, device, expected_classes=4)
    z_dark4_va, y2_va = logits_from_model(model_dark4, dl_va, device, expected_classes=4)
    z_dark4_te, y2_te = logits_from_model(model_dark4, dl_te, device, expected_classes=4)

    if not (np.array_equal(y_tr, y2_tr) and np.array_equal(y_va, y2_va) and np.array_equal(y_te, y2_te)):
        raise ValueError("ResNet2 与 DarkNet4 标签顺序不一致，无法逐样本融合。")

    p_dark_leak_tr, p_dark_no_leak_tr = parent_prob_candidates_from_dark4_logits(z_dark4_tr)
    p_dark_leak_va, p_dark_no_leak_va = parent_prob_candidates_from_dark4_logits(z_dark4_va)
    p_dark_leak_te, p_dark_no_leak_te = parent_prob_candidates_from_dark4_logits(z_dark4_te)

    val_f1_leak = bin_metrics_from_probs(p_dark_leak_va, y_va, thr=0.5)["f1"]
    val_f1_no_leak = bin_metrics_from_probs(p_dark_no_leak_va, y_va, thr=0.5)["f1"]

    if args.dark_parent_positive == "leak":
        parent_pos_mode = "leak(0/1)"
        p_dark_tr = p_dark_leak_tr
        p_dark_va = p_dark_leak_va
        p_dark_te = p_dark_leak_te
    elif args.dark_parent_positive == "no_leak":
        parent_pos_mode = "no_leak(2/3)"
        p_dark_tr = p_dark_no_leak_tr
        p_dark_va = p_dark_no_leak_va
        p_dark_te = p_dark_no_leak_te
    else:
        if val_f1_leak >= val_f1_no_leak:
            parent_pos_mode = "leak(0/1)"
            p_dark_tr = p_dark_leak_tr
            p_dark_va = p_dark_leak_va
            p_dark_te = p_dark_leak_te
        else:
            parent_pos_mode = "no_leak(2/3)"
            p_dark_tr = p_dark_no_leak_tr
            p_dark_va = p_dark_no_leak_va
            p_dark_te = p_dark_no_leak_te

    print(
        f"[ParentMapping] mode={parent_pos_mode} | "
        f"val_f1_if_leak={val_f1_leak:.4f} | val_f1_if_no_leak={val_f1_no_leak:.4f}"
    )

    z_dark2_tr = prob_to_2cls_logits_np(p_dark_tr)
    z_dark2_va = prob_to_2cls_logits_np(p_dark_va)
    z_dark2_te = prob_to_2cls_logits_np(p_dark_te)

    p_res_va = prob_pos_from_logits_np(z_res_va)
    p_res_te = prob_pos_from_logits_np(z_res_te)
    m_res_va = bin_metrics_from_probs(p_res_va, y_va, thr=0.5)
    m_dark_va = bin_metrics_from_probs(p_dark_va, y_va, thr=0.5)
    m_res_te = bin_metrics_from_probs(p_res_te, y_te, thr=0.5)
    m_dark_te = bin_metrics_from_probs(p_dark_te, y_te, thr=0.5)

    fusion = CWGFFusion2Cls(hid=args.fusion_hid, drop=args.fusion_drop).to(device)

    zr_tr_t = torch.tensor(z_res_tr, dtype=torch.float32, device=device)
    zd_tr_t = torch.tensor(z_dark2_tr, dtype=torch.float32, device=device)
    zr_va_t = torch.tensor(z_res_va, dtype=torch.float32, device=device)
    zd_va_t = torch.tensor(z_dark2_va, dtype=torch.float32, device=device)
    zr_te_t = torch.tensor(z_res_te, dtype=torch.float32, device=device)
    zd_te_t = torch.tensor(z_dark2_te, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    fusion.fit_feature_norm_from_logits(zr_tr_t, zd_tr_t)

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
        p_va_best, _ = fusion(zr_va_t, zd_va_t)
        p_te_best, _ = fusion(zr_te_t, zd_te_t)
        p_va_best_np = p_va_best.detach().cpu().numpy()
        p_te_best_np = p_te_best.detach().cpu().numpy()

    m_fuse_va = bin_metrics_from_probs(p_va_best_np, y_va, best_tau)
    m_fuse_te = bin_metrics_from_probs(p_te_best_np, y_te, best_tau)

    best_single_val_f1 = max(m_res_va["f1"], m_dark_va["f1"])
    degenerate = "degenerate" if (m_fuse_va["f1"] < best_single_val_f1) else "not degenerate"

    print(
        f"[VAL] ResNet2 f1={m_res_va['f1']:.4f} | Dark4-parent f1={m_dark_va['f1']:.4f} | "
        f"CWGF f1={m_fuse_va['f1']:.4f} => {degenerate}"
    )
    print(
        f"[TEST] ResNet2 f1={m_res_te['f1']:.4f} | Dark4-parent f1={m_dark_te['f1']:.4f} | "
        f"CWGF@tau={best_tau:.3f} f1={m_fuse_te['f1']:.4f}"
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = ""
    if bool(args.save_ckpt):
        ckpt_path = os.path.join(args.save_dir, f"fuse_cwgf_dark4parent_resnet2_best_{stamp}.pt")
        torch.save(
            {
                "method": "cwgf_dark4parent_resnet2",
                "state_dict": {k: v.cpu() for k, v in fusion.state_dict().items()},
                "phi_mu": fusion.phi_mu.detach().cpu(),
                "phi_std": fusion.phi_std.detach().cpu(),
                "best_tau": float(best_tau),
                "best_val_f1": float(best_val_f1),
                "parent_pos_mode": parent_pos_mode,
                "ckpt_resnet2": ckpt_res,
                "ckpt_darknet4": ckpt_dark4,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"[CKPT] saved: {ckpt_path}")

    log_path = os.path.join(args.log_dir, f"metrics_fuse_cwgf_dark4parent_resnet2_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "parent_positive_mode",
                "best_tau",
                "best_val_f1",
                "val_resnet_f1",
                "val_dark4_parent_f1",
                "val_fuse_f1",
                "degeneracy",
                "test_resnet_f1",
                "test_dark4_parent_f1",
                "test_fuse_f1",
            ]
        )
        w.writerow(
            [
                "cwgf_dark4parent_resnet2",
                parent_pos_mode,
                f"{best_tau:.6f}",
                f"{best_val_f1:.6f}",
                f"{m_res_va['f1']:.6f}",
                f"{m_dark_va['f1']:.6f}",
                f"{m_fuse_va['f1']:.6f}",
                degenerate,
                f"{m_res_te['f1']:.6f}",
                f"{m_dark_te['f1']:.6f}",
                f"{m_fuse_te['f1']:.6f}",
            ]
        )

    print(f"[LOG] saved: {log_path}")


def main():
    args = build_parser().parse_args()
    run_fusion(args)


if __name__ == "__main__":
    main()

