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


def freeze_model(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def load_resnet_2cls(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = safe_torch_load(ckpt_path, map_location="cpu")
    state = strip_module_prefix(ck.get("state_dict", ck))
    num_classes = int(ck.get("num_classes", 2))
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state, strict=False)
    model.to(device)
    freeze_model(model)
    return model


def _find_linear_dims_from_state_dict(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    w1 = state.get("net.0.weight", None)
    w2 = state.get("net.3.weight", None)
    if w1 is None or w2 is None:
        raise KeyError("Cannot infer CWT-MLP dims from checkpoint state_dict.")
    hidden = int(w1.shape[0])
    in_dim = int(w1.shape[1])
    if int(w2.shape[1]) != hidden:
        raise ValueError(f"Unexpected MLP2 shape: net.0={tuple(w1.shape)}, net.3={tuple(w2.shape)}")
    return in_dim, hidden


def load_cwt_mlp_2cls(ckpt_path: str, device: torch.device) -> nn.Module:
    ck = safe_torch_load(ckpt_path, map_location="cpu")
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
    return model


@torch.no_grad()
def logits_from_model(model: nn.Module, loader, device: torch.device, expected_classes: int):
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if out.ndim != 2 or out.shape[1] != expected_classes:
            raise ValueError(f"Expected [B,{expected_classes}] logits, got {tuple(out.shape)}")
        logits_all.append(out.detach().cpu().numpy().astype(np.float32))
        y_all.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0)


@torch.no_grad()
def logits_from_cwt_mlp(model: nn.Module, feats: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    outs = []
    n = feats.shape[0]
    for i in range(0, n, batch_size):
        x = torch.from_numpy(feats[i : i + batch_size]).to(device)
        logit_pos = model(x)
        logits = torch.stack([-logit_pos, logit_pos], dim=1)
        outs.append(logits.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0)


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
        ds_tr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    dl_va = torch.utils.data.DataLoader(
        ds_va, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    dl_te = torch.utils.data.DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    return dl_tr, dl_va, dl_te


def load_cwt_splits(data_root_cwt: str):
    paths = {
        "Xcwt_train": os.path.join(data_root_cwt, "Xcwt_train.npy"),
        "Xcwt_val": os.path.join(data_root_cwt, "Xcwt_val.npy"),
        "Xcwt_test": os.path.join(data_root_cwt, "Xcwt_test.npy"),
        "y_train": os.path.join(data_root_cwt, "y_train.npy"),
        "y_val": os.path.join(data_root_cwt, "y_val.npy"),
        "y_test": os.path.join(data_root_cwt, "y_test.npy"),
    }
    miss = [p for p in paths.values() if not os.path.exists(p)]
    if miss:
        raise FileNotFoundError("Missing CWT dataset files:\n" + "\n".join(miss))

    out = {k: np.load(v) for k, v in paths.items()}
    out["Xcwt_train"] = out["Xcwt_train"].astype(np.float32)
    out["Xcwt_val"] = out["Xcwt_val"].astype(np.float32)
    out["Xcwt_test"] = out["Xcwt_test"].astype(np.float32)
    out["y_train"] = out["y_train"].astype(np.int64)
    out["y_val"] = out["y_val"].astype(np.int64)
    out["y_test"] = out["y_test"].astype(np.int64)
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root_stft", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--data_root_cwt", type=str, default=cfg.DATASET_CWT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--ckpt_resnet2", type=str, default="")
    ap.add_argument("--ckpt_cwt_mlp2", type=str, default="")

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

    ckpt_res = resolve_latest_ckpt(args.ckpt_resnet2, "stft_resnet18_2cls_best_*.pth", "resnet2")
    ckpt_cwt = resolve_latest_ckpt(args.ckpt_cwt_mlp2, "mlp_cwt_2cls_best_*.pt", "cwt_mlp2")

    print("========== Fuse CWGF (ResNet18-2cls + CWT-MLP-2cls) ==========")
    print(f"device        : {device}")
    print(f"data_root_stft: {args.data_root_stft}")
    print(f"data_root_cwt : {args.data_root_cwt}")
    print(f"ckpt_resnet2  : {ckpt_res}")
    print(f"ckpt_cwt_mlp2 : {ckpt_cwt}")
    print("==============================================================")

    dl_tr, dl_va, dl_te = build_stft_loaders(args)
    model_res = load_resnet_2cls(ckpt_res, device)
    model_cwt = load_cwt_mlp_2cls(ckpt_cwt, device)

    z_res_tr, y_tr_stft = logits_from_model(model_res, dl_tr, device, expected_classes=2)
    z_res_va, y_va_stft = logits_from_model(model_res, dl_va, device, expected_classes=2)
    z_res_te, y_te_stft = logits_from_model(model_res, dl_te, device, expected_classes=2)

    cwt = load_cwt_splits(args.data_root_cwt)
    y_tr = cwt["y_train"]
    y_va = cwt["y_val"]
    y_te = cwt["y_test"]
    if not (np.array_equal(y_tr, y_tr_stft) and np.array_equal(y_va, y_va_stft) and np.array_equal(y_te, y_te_stft)):
        raise ValueError("CWT 与 STFT 标签顺序不一致，无法逐样本融合。")

    z_cwt_tr = logits_from_cwt_mlp(model_cwt, cwt["Xcwt_train"], args.batch_size, device)
    z_cwt_va = logits_from_cwt_mlp(model_cwt, cwt["Xcwt_val"], args.batch_size, device)
    z_cwt_te = logits_from_cwt_mlp(model_cwt, cwt["Xcwt_test"], args.batch_size, device)

    p_res_va = prob_pos_from_logits_np(z_res_va)
    p_res_te = prob_pos_from_logits_np(z_res_te)
    p_cwt_va = prob_pos_from_logits_np(z_cwt_va)
    p_cwt_te = prob_pos_from_logits_np(z_cwt_te)

    tau_res, _ = find_best_thr(
        p_res_va, y_va, lo=float(args.thr_lo), hi=float(args.thr_hi), step=float(args.thr_step), objective=str(args.thr_objective)
    )
    tau_cwt, _ = find_best_thr(
        p_cwt_va, y_va, lo=float(args.thr_lo), hi=float(args.thr_hi), step=float(args.thr_step), objective=str(args.thr_objective)
    )
    m_res_va = bin_metrics_from_probs(p_res_va, y_va, tau_res)
    m_cwt_va = bin_metrics_from_probs(p_cwt_va, y_va, tau_cwt)
    m_res_te = bin_metrics_from_probs(p_res_te, y_te, tau_res)
    m_cwt_te = bin_metrics_from_probs(p_cwt_te, y_te, tau_cwt)

    fusion = CWGFFusion2Cls(hid=int(args.fusion_hid), drop=float(args.fusion_drop)).to(device)

    zr_tr_t = torch.tensor(z_res_tr, dtype=torch.float32, device=device)
    zc_tr_t = torch.tensor(z_cwt_tr, dtype=torch.float32, device=device)
    zr_va_t = torch.tensor(z_res_va, dtype=torch.float32, device=device)
    zc_va_t = torch.tensor(z_cwt_va, dtype=torch.float32, device=device)
    zr_te_t = torch.tensor(z_res_te, dtype=torch.float32, device=device)
    zc_te_t = torch.tensor(z_cwt_te, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)

    fusion.fit_feature_norm_from_logits(zr_tr_t, zc_tr_t)

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
        p_tr, aux_tr = fusion(zr_tr_t, zc_tr_t)
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
                p_va = fusion(zr_va_t, zc_va_t)[0].detach().cpu().numpy()
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
        p_va_fuse = fusion(zr_va_t, zc_va_t)[0].detach().cpu().numpy()
        p_te_fuse = fusion(zr_te_t, zc_te_t)[0].detach().cpu().numpy()
    m_fuse_va = bin_metrics_from_probs(p_va_fuse, y_va, best_tau)
    m_fuse_te = bin_metrics_from_probs(p_te_fuse, y_te, best_tau)

    best_single_val_f1 = max(m_res_va["f1"], m_cwt_va["f1"])
    degenerate = "degenerate" if (m_fuse_va["f1"] < best_single_val_f1) else "not degenerate"

    print(
        f"[VAL] ResNet2 f1={m_res_va['f1']:.4f} (tau={tau_res:.3f}) | "
        f"CWT-MLP2 f1={m_cwt_va['f1']:.4f} (tau={tau_cwt:.3f}) | "
        f"CWGF f1={m_fuse_va['f1']:.4f} (tau={best_tau:.3f}) => {degenerate}"
    )
    print(
        f"[TEST] ResNet2 f1={m_res_te['f1']:.4f} | CWT-MLP2 f1={m_cwt_te['f1']:.4f} | "
        f"CWGF f1={m_fuse_te['f1']:.4f}"
    )

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if bool(args.save_ckpt):
        ckpt_path = os.path.join(args.save_dir, f"fuse_cwgf_resnet2_cwtmlp2_2cls_{stamp}.pt")
        torch.save(
            {
                "model": "CWGF_RESNET2_CWTMLP2_2CLS",
                "state_dict": fusion.state_dict(),
                "best_thr": best_tau,
                "best_val_f1": best_val_f1,
                "seed": int(args.seed),
            },
            ckpt_path,
        )
        print(f"[CKPT] {ckpt_path}")

    log_path = os.path.join(args.log_dir, f"metrics_fuse_cwgf_resnet2_cwtmlp2_2cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "method",
                "val_best_f1",
                "val_best_acc",
                "val_best_thr",
                "val_resnet_f1",
                "val_resnet_thr",
                "val_cwt_mlp_f1",
                "val_cwt_mlp_thr",
                "val_fuse_f1",
                "degeneracy",
                "test_resnet_f1",
                "test_cwt_mlp_f1",
                "test_fuse_f1",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "method": "cwgf_resnet2_cwtmlp2_2cls",
                "val_best_f1": f"{best_val_f1:.6f}",
                "val_best_acc": f"{best_val_acc:.6f}",
                "val_best_thr": f"{best_tau:.6f}",
                "val_resnet_f1": f"{m_res_va['f1']:.6f}",
                "val_resnet_thr": f"{tau_res:.6f}",
                "val_cwt_mlp_f1": f"{m_cwt_va['f1']:.6f}",
                "val_cwt_mlp_thr": f"{tau_cwt:.6f}",
                "val_fuse_f1": f"{m_fuse_va['f1']:.6f}",
                "degeneracy": degenerate,
                "test_resnet_f1": f"{m_res_te['f1']:.6f}",
                "test_cwt_mlp_f1": f"{m_cwt_te['f1']:.6f}",
                "test_fuse_f1": f"{m_fuse_te['f1']:.6f}",
            }
        )
    print(f"[LOG] {log_path}")


def main():
    args = build_parser().parse_args()
    run_fusion(args)


if __name__ == "__main__":
    main()
