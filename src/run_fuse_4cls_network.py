import os, csv, argparse, random
import numpy as np
from datetime import datetime
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, confusion_matrix

import config as cfg
from NpyDataset import NpyDataset          # STFT 图像 → Tensor（imagenet 归一化）
from Darknet19 import Darknet19            # 你的 STFT 4类专家

class MLP4(nn.Module):
    def __init__(self, in_dim, hidden=128, p_drop=0.2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes)
        )
    def forward(self, x): return self.net(x)

class CWTDataset(torch.utils.data.Dataset):
    """CWT 紧凑向量 + y4 标签"""
    def __init__(self, X_path: str, y4_path: str):
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y4_path).astype(np.int64)
        assert len(self.X) == len(self.y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)

# —— 逐类 Gate：沿用你的 Gatenet 结构，但按类共享、逐类输出 1 维权重 ——
from Gatenet import Gatenet  # in_dim=5, hid=16, 输出标量∈(0,1)

# ========== Utils ==========
def set_seeds(seed=2025):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def logits_from_model_4cls(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all, y_all = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        z = model(x)                           # [B,4] logits
        logits_all.append(z.cpu().numpy())
        y_all.append(y.numpy().astype(np.int64))
    return np.concatenate(logits_all, 0), np.concatenate(y_all, 0)

def probs_from_logits_np(z: np.ndarray) -> np.ndarray:
    z_max = z.max(axis=1, keepdims=True)
    ez = np.exp(z - z_max)
    return ez / ez.sum(axis=1, keepdims=True)  # [N,4]

def entropy_categorical_np(P: np.ndarray, eps=1e-9) -> np.ndarray:
    P = np.clip(P, eps, 1.0)
    return -(P * np.log(P)).sum(axis=1)        # [N]

def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))

def macro_rec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(recall_score(y_true, y_pred, average="macro", zero_division=0))

def parent_from_y4(y4: np.ndarray, pos_idx=(0,1)) -> np.ndarray:
    mask = np.zeros_like(y4, dtype=np.int64)
    for k in pos_idx: mask |= (y4 == k)
    return mask  # 1=漏, 0=非漏

def parent_from_pred4(y4_pred: np.ndarray, pos_idx=(0,1)) -> np.ndarray:
    return parent_from_y4(y4_pred, pos_idx)

def metrics_4cls_and_parent(P: np.ndarray, y4: np.ndarray, pos_idx=(0,1)):
    y4_pred = P.argmax(axis=1)
    mf1  = macro_f1(y4, y4_pred)
    mrec = macro_rec(y4, y4_pred)
    # 父级
    yb_true = parent_from_y4(y4, pos_idx)
    yb_pred = parent_from_pred4(y4_pred, pos_idx)
    bf1  = float(f1_score(yb_true, yb_pred, average="binary", zero_division=0))
    brec = float(recall_score(yb_true, yb_pred, average="binary", zero_division=0))
    cm4  = confusion_matrix(y4, y4_pred, labels=[0,1,2,3])
    return {"macro_f1": mf1, "macro_rec": mrec, "bin_f1": bf1, "bin_rec": brec, "cm4": cm4}

def build_per_class_features(Pd: np.ndarray, Pm: np.ndarray) -> np.ndarray:
    """
    构造逐类 gate 特征：shape [N,4,5]
      [ p_dark_k, p_mlp_k, |Δ_k|, H_dark, H_mlp ]
    """
    assert Pd.shape == Pm.shape and Pd.ndim == 2 and Pd.shape[1] == 4
    N = Pd.shape[0]
    Hd = entropy_categorical_np(Pd)           # [N]
    Hm = entropy_categorical_np(Pm)           # [N]
    Hd4 = np.repeat(Hd[:,None], 4, axis=1)    # [N,4]
    Hm4 = np.repeat(Hm[:,None], 4, axis=1)    # [N,4]
    feats = np.stack([Pd, Pm, np.abs(Pd - Pm), Hd4, Hm4], axis=2)  # [N,4,5]
    return feats.astype(np.float32)

def norm_feats(feats_tr: np.ndarray, feats: np.ndarray):
    # 按特征维度做标准化（在训练集 [N*4,5] 上统计）
    mu = feats_tr.reshape(-1, feats_tr.shape[-1]).mean(axis=0, keepdims=True)   # [1,5]
    sg = feats_tr.reshape(-1, feats_tr.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    f_n = (feats - mu) / sg
    return f_n.astype(np.float32), mu, sg

def init_gate_half(gate: nn.Module):
    linears = [m for m in gate.modules() if isinstance(m, nn.Linear)]
    assert len(linears) >= 1, "Gatenet 里至少要有一层 nn.Linear"
    for m in linears[:-1]:
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    last = linears[-1]
    nn.init.zeros_(last.weight)
    if last.bias is not None:
        nn.init.zeros_(last.bias)

def fuse_logits_per_class(z_dark: torch.Tensor, z_mlp: torch.Tensor, g4: torch.Tensor):
    """
    建议用“逐类几何平均”的 log-prob 融合：
      w_k = g_k * log_softmax(z_dark)_k + (1-g_k) * log_softmax(z_mlp)_k
    返回的 w 直接可作为 CrossEntropy 的 logits（内部会再做 logsumexp 归一化）。
    """
    ld = F.log_softmax(z_dark, dim=1)   # [N,4] 对数概率
    lm = F.log_softmax(z_mlp,  dim=1)   # [N,4]
    return g4 * ld + (1.0 - g4) * lm    # [N,4] 逐类加权的“对数几率”

# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    # 数据根目录
    ap.add_argument("--data_root_stft", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--data_root_cwt",  type=str, default=cfg.DATASET_CWT)
    ap.add_argument("--img_size",       type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size",     type=int, default=512)

    # 专家 ckpt
    ap.add_argument("--ckpt_darknet4",  type=str, default="./checkpoints/darknet19_4cls_best.pth")
    ap.add_argument("--ckpt_mlp4",      type=str, default="./checkpoints/mlp_cwt_4cls_best.pt")

    # Gate 训练
    ap.add_argument("--gate_hid",       type=int, default=16)
    ap.add_argument("--gate_epochs",    type=int, default=200)
    ap.add_argument("--gate_lr",        type=float, default=1e-4)
    ap.add_argument("--gate_wd",        type=float, default=1e-3)
    ap.add_argument("--gate_center",    type=float, default=1e-3)  # (g-0.5)^2
    ap.add_argument("--seed",           type=int,   default=2025)
    args = ap.parse_args()

    set_seeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- STFT: Darknet19 ----------
    Xtr_s = os.path.join(args.data_root_stft, f"X_train_stft_{args.img_size}.npy")
    Xva_s = os.path.join(args.data_root_stft, f"X_val_stft_{args.img_size}.npy")
    Xte_s = os.path.join(args.data_root_stft, f"X_test_stft_{args.img_size}.npy")
    y4tr_s = os.path.join(args.data_root_stft, "y4_train.npy")
    y4va_s = os.path.join(args.data_root_stft, "y4_val.npy")
    y4te_s = os.path.join(args.data_root_stft, "y4_test.npy")
    ds_s_tr = NpyDataset(Xtr_s, y4tr_s, normalize="imagenet", memmap=True)
    ds_s_va = NpyDataset(Xva_s, y4va_s, normalize="imagenet", memmap=True)
    ds_s_te = NpyDataset(Xte_s, y4te_s, normalize="imagenet", memmap=True)
    dl_s_tr = torch.utils.data.DataLoader(ds_s_tr, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_s_va = torch.utils.data.DataLoader(ds_s_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_s_te = torch.utils.data.DataLoader(ds_s_te, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ---------- CWT: MLP4 ----------
    Xtr_c = os.path.join(args.data_root_cwt, "Xcwt_train.npy")
    Xva_c = os.path.join(args.data_root_cwt, "Xcwt_val.npy")
    Xte_c = os.path.join(args.data_root_cwt, "Xcwt_test.npy")
    y4tr_c = os.path.join(args.data_root_cwt, "y4_train.npy")
    y4va_c = os.path.join(args.data_root_cwt, "y4_val.npy")
    y4te_c = os.path.join(args.data_root_cwt, "y4_test.npy")
    # 维度
    in_dim_cwt = int(np.load(Xtr_c, mmap_mode="r").shape[1])
    ds_c_tr = CWTDataset(Xtr_c, y4tr_c)
    ds_c_va = CWTDataset(Xva_c, y4va_c)
    ds_c_te = CWTDataset(Xte_c, y4te_c)
    dl_c_tr = torch.utils.data.DataLoader(ds_c_tr, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    dl_c_va = torch.utils.data.DataLoader(ds_c_va, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    dl_c_te = torch.utils.data.DataLoader(ds_c_te, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # ---------- 加载专家 ----------
    # Darknet19
    model_dark = Darknet19(num_classes=4).to(device)
    sd_dark = torch.load(args.ckpt_darknet4, map_location="cpu")
    sd_dark = sd_dark.get("state_dict", sd_dark)
    sd_dark = { (k[7:] if k.startswith("module.") else k): v for k, v in sd_dark.items() }
    model_dark.load_state_dict(sd_dark, strict=False)
    model_dark.eval()

    # MLP4
    model_mlp = MLP4(in_dim=in_dim_cwt, hidden=128, p_drop=0.2, num_classes=4).to(device)
    sd_mlp = torch.load(args.ckpt_mlp4, map_location="cpu")
    sd_mlp = sd_mlp.get("state_dict", sd_mlp)
    model_mlp.load_state_dict(sd_mlp, strict=True)
    model_mlp.eval()

    # ---------- 取 logits（四类） ----------
    zD_tr, y4_tr = logits_from_model_4cls(model_dark, dl_s_tr, device)
    zD_va, y4_va = logits_from_model_4cls(model_dark, dl_s_va, device)
    zD_te, y4_te = logits_from_model_4cls(model_dark, dl_s_te, device)

    zM_tr, y4c_tr = logits_from_model_4cls(model_mlp,  dl_c_tr, device)
    zM_va, y4c_va = logits_from_model_4cls(model_mlp,  dl_c_va, device)
    zM_te, y4c_te = logits_from_model_4cls(model_mlp,  dl_c_te, device)

    # 对齐性检查
    if not (np.array_equal(y4_tr, y4c_tr) and np.array_equal(y4_va, y4c_va) and np.array_equal(y4_te, y4c_te)):
        raise ValueError("CWT 与 STFT 的 y4 顺序/划分不一致，无法一一对应融合。")

    # 概率与逐类特征
    PD_tr, PD_va, PD_te = probs_from_logits_np(zD_tr), probs_from_logits_np(zD_va), probs_from_logits_np(zD_te)
    PM_tr, PM_va, PM_te = probs_from_logits_np(zM_tr), probs_from_logits_np(zM_va), probs_from_logits_np(zM_te)
    feat_tr = build_per_class_features(PD_tr, PM_tr)   # [N,4,5]
    feat_va = build_per_class_features(PD_va, PM_va)
    feat_te = build_per_class_features(PD_te, PM_te)
    feat_tr_n, mu_phi, sg_phi = norm_feats(feat_tr, feat_tr)
    feat_va_n = (feat_va - mu_phi) / sg_phi
    feat_te_n = (feat_te - mu_phi) / sg_phi

    # ---------- Gate：逐类共享单元 ----------
    gate = Gatenet(in_dim=5, hid=args.gate_hid, drop=0.1).to(device)
    init_gate_half(gate)  # 初始输出≈0.5

    opt = torch.optim.AdamW(gate.parameters(), lr=args.gate_lr, weight_decay=args.gate_wd)
    ce  = nn.CrossEntropyLoss()

    # 变成 Tensor
    ZD_tr = torch.tensor(zD_tr, dtype=torch.float32, device=device)  # [N,4]
    ZM_tr = torch.tensor(zM_tr, dtype=torch.float32, device=device)
    ZD_va = torch.tensor(zD_va, dtype=torch.float32, device=device)
    ZM_va = torch.tensor(zM_va, dtype=torch.float32, device=device)
    Y_tr  = torch.tensor(y4_tr, dtype=torch.long, device=device)
    Y_va  = torch.tensor(y4_va, dtype=torch.long, device=device)

    F_tr = torch.tensor(feat_tr_n, dtype=torch.float32, device=device) # [N,4,5]
    F_va = torch.tensor(feat_va_n, dtype=torch.float32, device=device)

    best_va_mf1, best_state = -1.0, None

    for ep in range(1, args.gate_epochs + 1):
        gate.train(); opt.zero_grad()
        # 逐类共享：把 [N,4,5] 拉平成 [N*4,5]，过 gate，再 reshape 回 [N,4]
        g_tr = gate(F_tr.reshape(-1, F_tr.shape[-1])).reshape(-1, 4)     # [N,4] in (0,1)
        Zf_tr = fuse_logits_per_class(ZD_tr, ZM_tr, g_tr)                # [N,4]
        loss  = ce(Zf_tr, Y_tr) + args.gate_center * (g_tr - 0.5).pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()

        # 验证选最优 epoch（Macro-F1）
        if ep % 10 == 0 or ep == 1:
            gate.eval()
            with torch.no_grad():
                g_va  = gate(F_va.reshape(-1, F_va.shape[-1])).reshape(-1, 4)
                Zf_va = fuse_logits_per_class(ZD_va, ZM_va, g_va)
                Pf_va = F.softmax(Zf_va, dim=1).cpu().numpy()
                mets  = metrics_4cls_and_parent(Pf_va, y4_va, pos_idx=(0,1))
                if mets["macro_f1"] > best_va_mf1:
                    best_va_mf1 = mets["macro_f1"]
                    best_state = {k: v.detach().cpu() for k, v in gate.state_dict().items()}
            print(f"Epoch {ep:03d} | gate_loss={loss.item():.4f} | val_macroF1={mets['macro_f1']:.4f} "
                  f"| best={best_va_mf1:.4f}")

    if best_state is not None:
        gate.load_state_dict(best_state, strict=True)
    gate.eval()

    # ---------- TEST ----------
    with torch.no_grad():
        F_te = torch.tensor(feat_te_n, dtype=torch.float32, device=device)  # [N,4,5]
        g_te  = gate(F_te.reshape(-1, F_te.shape[-1])).reshape(-1, 4)
        ZD_te = torch.tensor(zD_te, dtype=torch.float32, device=device)
        ZM_te = torch.tensor(zM_te, dtype=torch.float32, device=device)
        Zf_te = fuse_logits_per_class(ZD_te, ZM_te, g_te)
        Pf_te = F.softmax(Zf_te, dim=1).cpu().numpy()

    # 两位专家各自四分类指标（公平对比，直接 argmax）
    PD_te = probs_from_logits_np(zD_te)
    PM_te = probs_from_logits_np(zM_te)
    m_fuse = metrics_4cls_and_parent(Pf_te, y4_te, pos_idx=(0,1))
    m_dark = metrics_4cls_and_parent(PD_te, y4_te, pos_idx=(0,1))
    m_mlp  = metrics_4cls_and_parent(PM_te, y4_te, pos_idx=(0,1))

    print("\n[FUSE-4CLS] TEST | Macro-F1={:.4f} Macro-Rec={:.4f} | Bin-F1={:.4f} Bin-Rec={:.4f}".format(
        m_fuse["macro_f1"], m_fuse["macro_rec"], m_fuse["bin_f1"], m_fuse["bin_rec"]))
    print("[FUSE] Confusion Matrix (4x4, rows=true, cols=pred):\n", m_fuse["cm4"])

    print("\n[BASE] DarkNet-19  | Macro-F1={:.4f} Bin-F1={:.4f}".format(
        m_dark["macro_f1"], m_dark["bin_f1"]))
    print("[BASE] MLP (CWT)   | Macro-F1={:.4f} Bin-F1={:.4f}".format(
        m_mlp["macro_f1"], m_mlp["bin_f1"]))

    # ---------- 保存 ----------
    os.makedirs(cfg.CKPT_DIR, exist_ok=True); os.makedirs(cfg.LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(cfg.CKPT_DIR, f"fuse4_mlp_cwt__darknet_stft_{stamp}.pt")
    torch.save({
        "model": "GateNet4-per-class",
        "state_dict": {k: v.cpu() for k, v in gate.state_dict().items()},
        "phi_mu": mu_phi, "phi_std": sg_phi,
        "best_val_macro_f1": best_va_mf1
    }, ckpt_path)
    print(f"[CKPT] gate saved: {ckpt_path}")

    log_path = os.path.join(cfg.LOG_DIR, f"metrics_fuse4_mlp_darknet_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["macro_f1","macro_rec","bin_f1","bin_rec"])
        w.writerow([f"{m_fuse['macro_f1']:.6f}", f"{m_fuse['macro_rec']:.6f}",
                    f"{m_fuse['bin_f1']:.6f}",  f"{m_fuse['bin_rec']:.6f}"])
    print(f"[LOG] saved: {log_path}")

if __name__ == "__main__":
    main()
