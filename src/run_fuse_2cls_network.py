import os, csv, argparse
import numpy as np
import torch
import torch.nn as nn
import config as cfg

from datetime import datetime
from Darknet19 import Darknet19
from HierarchicalDarknet19 import HierarchicalDarknet19
from Gatenet import Gatenet
from NpyDataset import NpyDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

def entropy01_np(p, eps=1e-9):
    p = np.clip(p, eps, 1.0 - eps)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))

def entropy_reg(p, eps=1e-9):
    p = torch.clamp(p, eps, 1-eps)
    return (p * torch.log(p) + (1-p) * torch.log(1-p)).mean()

def entropy_pos(p, eps=1e-9):
    p = torch.clamp(p, eps, 1-eps)
    return ( - p*torch.log(p) - (1-p)*torch.log(1-p) ).mean()

def make_phi(pr, pc):
    return np.stack([pr, pc, np.abs(pr-pc), entropy01_np(pr), entropy01_np(pc)], axis=1).astype(np.float32)

def bin_metrics_from_probs(p: np.ndarray, y: np.ndarray, thr: float):
    if p.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch in bin_metrics_from_probs: p has {p.shape[0]} samples, y has {y.shape[0]} samples.")
    y_hat = (p >= thr).astype(np.int64)
    cm = confusion_matrix(y, y_hat, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average='binary', zero_division=0)
    bal_acc = 0.5 * ( (tp / max(tp+fn,1)) + (tn / max(tn+fp,1)) )
    return {"acc":acc, "precision":prec, "recall":rec, "f1":f1, "bal_acc":bal_acc, "cm":cm}

def find_best_thr(p, y, lo=0.01, hi=0.99, step=0.01, objective="f1"):
    # 1) 用 linspace 生成阈值，避免累加误差；四舍五入到 1e-6
    n = int(np.floor((hi - lo) / step)) + 1
    taus = np.round(np.linspace(lo, hi, n), 6)

    def _score(t):
        m = bin_metrics_from_probs(p, y, t)
        return m["f1"] if objective == "f1" else m["acc"]

    # 2) 粗扫
    scores = np.array([_score(t) for t in taus])
    i = int(np.argmax(scores))
    tau1 = float(taus[i])

    # 3) 细扫（在 tau1±5*step 的窗口内缩小 10 倍步长）
    lo2 = max(lo, tau1 - 5*step)
    hi2 = min(hi, tau1 + 5*step)
    step2 = step / 10.0
    n2 = int(np.floor((hi2 - lo2) / step2)) + 1
    taus2 = np.round(np.linspace(lo2, hi2, n2), 6)
    scores2 = np.array([_score(t) for t in taus2])
    j = int(np.argmax(scores2))
    return float(taus2[j]), float(scores2[j])

def bce_on_probs(p, y, w_pos=1.0, w_neg=1.0, eps=1e-6):
    p = torch.clamp(p, eps, 1-eps)
    return ( - w_pos*y*torch.log(p) - w_neg*(1-y)*torch.log(1-p) ).mean()

def init_gate_equal(gate: nn.Module, small: float = 1e-3):
    last_lin = None
    for m in gate.modules():
        if isinstance(m, nn.Linear):
            last_lin = m
    if last_lin is None:
        raise RuntimeError("网络内未找到 nn.Linear，无法做等权初始化。")

    with torch.no_grad():
        nn.init.normal_(last_lin.weight, mean=0.0, std=small)
        if last_lin.bias is not None:
            last_lin.bias.zero_()

def logit_t(x, eps=1e-6):
    x = torch.clamp(x, eps, 1.0 - eps)
    return torch.log(x) - torch.log(1.0 - x)

@torch.no_grad()
def probs_from_model(model, loader, device, map_4cls_to_pos_idx=None, eps=1e-9):
    """
    Generic: returns prob_positive for each example (numpy), and true labels
    """
    model.eval()
    probs = []
    ys = []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        if torch.is_tensor(out) and out.dim() == 1:
            p = torch.sigmoid(out).detach().cpu().numpy()
        elif torch.is_tensor(out) and out.dim() == 2:
            logits = out.detach()
            probs_soft = torch.softmax(logits, dim=1).cpu().numpy()  # [B,C]
            C = probs_soft.shape[1]
            if C == 2:
                p = probs_soft[:, 1]
            else:
                if map_4cls_to_pos_idx is None:
                    p = probs_soft[:, -1]
                else:
                    p = probs_soft[:, map_4cls_to_pos_idx].sum(axis=1)
        else:
            p = torch.sigmoid(out).detach().cpu().numpy()
        # clip numerics to avoid exact 0/1
        p = np.clip(p, eps, 1.0 - eps)
        probs.append(p)
        ys.append(y.numpy().astype(np.int64))
    probs = np.concatenate(probs, axis=0)
    ys = np.concatenate(ys, axis=0)
    return probs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--ckpt_2cls", type=str, default="./checkpoints/darknet19_best.pth")
    ap.add_argument("--ckpt_4cls", type=str, default="./checkpoints/darknet19_hier_best.pth")
    ap.add_argument("--gate_hid", type=int, default=16)
    ap.add_argument("--gate_epochs", type=int, default=200)
    ap.add_argument("--gate_lr", type=float, default=1e-4)
    ap.add_argument("--thr_lo", type=float, default=0.20)
    ap.add_argument("--thr_hi", type=float, default=0.80)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--pos_pow", type=float, default=0.2)
    ap.add_argument("--map_pos_idx", type=str, default="0,1")
    args = ap.parse_args()

    map_pos_idx = [int(x) for x in args.map_pos_idx.split(",")]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = args.data_root

    X_tr = os.path.join(root, f"X_train_stft_{args.img_size}.npy")
    X_va = os.path.join(root, f"X_val_stft_{args.img_size}.npy")
    X_te = os.path.join(root, f"X_test_stft_{args.img_size}.npy")
    y_tr = os.path.join(root, "y_train.npy")
    y_va = os.path.join(root, "y_val.npy")
    y_te = os.path.join(root, "y_test.npy")

    if not(os.path.exists(X_tr) and os.path.exists(X_va) and os.path.exists(X_te) and
           os.path.exists(y_tr) and os.path.exists(y_va) and os.path.exists(y_te)):
        raise FileNotFoundError("Required dataset files not found in data_root. See script docstring.")

    ds_r_tr = NpyDataset(X_tr, y_tr, normalize="imagenet", memmap=True)
    ds_r_va = NpyDataset(X_va, y_va, normalize="imagenet", memmap=True)
    ds_r_te = NpyDataset(X_te, y_te, normalize="imagenet", memmap=True)

    ds_c_tr = NpyDataset(X_tr, y_tr, normalize="imagenet", memmap=True)
    ds_c_va = NpyDataset(X_va, y_va, normalize="imagenet", memmap=True)
    ds_c_te = NpyDataset(X_te, y_te, normalize="imagenet", memmap=True)

    dl_r_tr = torch.utils.data.DataLoader(ds_r_tr, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_r_va = torch.utils.data.DataLoader(ds_r_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_r_te = torch.utils.data.DataLoader(ds_r_te, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    dl_c_tr = torch.utils.data.DataLoader(ds_c_tr, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_c_va = torch.utils.data.DataLoader(ds_c_va, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    dl_c_te = torch.utils.data.DataLoader(ds_c_te, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    ck2 = torch.load(args.ckpt_2cls, map_location="cpu")
    state2 = ck2.get("state_dict", ck2)
    num_classes_2 = int(ck2.get("num_classes", 2))
    model2 = Darknet19(num_classes=num_classes_2)
    new_state2 = {}
    for k,v in state2.items():
        if k.startswith("module."):
            new_state2[k[7:]] = v
        else:
            new_state2[k] = v
    model2.load_state_dict(new_state2, strict=False)
    model2.to(device); model2.eval()

    ck4 = torch.load(args.ckpt_4cls, map_location="cpu")
    state4 = ck4.get("state_dict", ck4)
    num_classes_4 = int(ck4.get("num_classes", 4))
    model4 = HierarchicalDarknet19(num_classes=num_classes_4).to(device)
    new_state4 = {}
    for k,v in state4.items():
        if k.startswith("module."):
            new_state4[k[7:]] = v
        else:
            new_state4[k] = v
    model4.load_state_dict(new_state4, strict=False)
    model4.eval()

    pr_tr, y_tr_arr = probs_from_model(model2, dl_r_tr, device, map_4cls_to_pos_idx=None)
    pr_va, _ = probs_from_model(model2, dl_r_va, device, map_4cls_to_pos_idx=None)
    pr_te, _ = probs_from_model(model2, dl_r_te, device, map_4cls_to_pos_idx=None)

    @torch.no_grad()
    def probs_from_4cls_model(hmodel, loader, device, pos_idx_list, eps=1e-9):
        hmodel.eval()
        p_list = []
        y_list = []
        for x, y in loader:
            x = x.to(device)
            out = hmodel(x)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                child_logits = out[1]
            else:
                child_logits = out
            probs_soft = torch.softmax(child_logits, dim=1).cpu().numpy()
            p_pos = probs_soft[:, pos_idx_list].sum(axis=1)
            p_pos = np.clip(p_pos, eps, 1.0 - eps)
            p_list.append(p_pos)
            y_list.append(y.numpy().astype(np.int64))
        return np.concatenate(p_list, axis=0), np.concatenate(y_list, axis=0)

    pc_tr, _ = probs_from_4cls_model(model4, dl_c_tr, device, map_pos_idx)
    pc_va, _ = probs_from_4cls_model(model4, dl_c_va, device, map_pos_idx)
    pc_te, _ = probs_from_4cls_model(model4, dl_c_te, device, map_pos_idx)

    # quick shape/nan checks
    y_val_arr = np.load(os.path.join(root, "y_val.npy"))
    print("Shapes: pr_va, pc_va, y_val =", pr_va.shape, pc_va.shape, y_val_arr.shape)
    if pr_va.shape[0] != y_val_arr.shape[0] or pc_va.shape[0] != y_val_arr.shape[0]:
        raise ValueError("Validation probabilities and y_val length mismatch. Check dataloaders.")
    
    m_r = bin_metrics_from_probs(pr_va, y_val_arr, 0.5)
    m_c = bin_metrics_from_probs(pc_va, y_val_arr, 0.5)
    print(f"[BASE] Darknet(2cls) val@{0.5:.2f}: acc={m_r['acc']:.4f} f1={m_r['f1']:.4f}")
    print(f"[BASE] Hier(4cls->bin) val@{0.5:.2f}: acc={m_c['acc']:.4f} f1={m_c['f1']:.4f}")

    phi_tr = make_phi(pr_tr, pc_tr)
    phi_va = make_phi(pr_va, pc_va)
    phi_te = make_phi(pr_te, pc_te)
    phi_tr = np.nan_to_num(phi_tr, nan=0.0, posinf=1.0, neginf=0.0)
    phi_va = np.nan_to_num(phi_va, nan=0.0, posinf=1.0, neginf=0.0)
    phi_te = np.nan_to_num(phi_te, nan=0.0, posinf=1.0, neginf=0.0)

    mu_phi = phi_tr.mean(axis=0, keepdims=True)
    std_phi = phi_tr.std(axis=0, keepdims=True) + 1e-6
    phi_tr_n = (phi_tr - mu_phi) / std_phi
    phi_va_n = (phi_va - mu_phi) / std_phi
    phi_te_n = (phi_te - mu_phi) / std_phi

    # 训练
    gate = Gatenet(in_dim=5, hid=args.gate_hid).to(device)
    init_gate_equal(gate, small=1e-3)
    opt = torch.optim.Adam(gate.parameters(), lr=args.gate_lr, weight_decay=1e-3)

    y_tr_t = torch.tensor(y_tr_arr, dtype=torch.float32, device=device)
    Zv_tr_t = torch.tensor(phi_tr_n, dtype=torch.float32, device=device)
    Pr_tr_t = torch.tensor(pr_tr, dtype=torch.float32, device=device)
    Pc_tr_t = torch.tensor(pc_tr, dtype=torch.float32, device=device)

    n1 = int((y_tr_t.cpu().numpy() == 1).sum())
    n0 = int((y_tr_t.cpu().numpy() == 0).sum())
    pos = float((y_tr_arr == 1).sum()); neg = float((y_tr_arr == 0).sum())
    w1 = neg / max(pos, 1.0)
    w0 = 1.0
    print(f"[GATE] pos_weight={w1:.3f} (pow={args.pos_pow})")

    best_f1, best_tau, best_state = -1.0, 0.5, None
    eps_clamp = 1e-6

    # 验证集用于 early stop / best_f1
    Zv_va_t = torch.tensor(phi_va_n, dtype=torch.float32, device=device)
    yv_va_t = torch.tensor(y_val_arr, dtype=torch.float32, device=device)

    pr_va_t = torch.tensor(pr_va, dtype=torch.float32, device=device)
    pc_va_t = torch.tensor(pc_va, dtype=torch.float32, device=device)

    for ep in range(1, args.gate_epochs + 1):
        gate.train()
        opt.zero_grad()
        g_tr = gate(Zv_tr_t)
        p_fuse_tr = torch.sigmoid(g_tr * logit_t(Pr_tr_t) + (1.0 - g_tr) * logit_t(Pc_tr_t))
        lambda_ent = 0.1   # try 0.05, 0.1, 0.2
        bce = bce_on_probs(p_fuse_tr, y_tr_t, w_pos=w1, w_neg=w0)
        lambda_gprior = 1e-3  # 可在 [1e-4,5e-3] 里试
        loss = bce - lambda_ent * entropy_pos(p_fuse_tr) + lambda_gprior * (g_tr - 0.5).pow(2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
        opt.step()

        if ep % 20 == 0 or ep == 1:
            gate.eval()
            with torch.no_grad():
                # 用验证集计算 f1
                g_va = gate(Zv_va_t)
                p_fuse_va = torch.sigmoid(g_va * logit_t(pr_va_t) + (1.0 - g_va) * logit_t(pc_va_t))

                # 转 numpy 供 find_best_thr 使用
                p_fuse_va_np = p_fuse_va.detach().cpu().numpy()
                tau, f1_ = find_best_thr(p_fuse_va_np, y_val_arr,
                                        args.thr_lo, args.thr_hi, args.thr_step)
                
                print(f"Epoch {ep:03d} | gate_loss={loss.item():.4f} | val@{tau:.2f} f1={f1_:.4f} | best@{best_tau:.2f} f1={best_f1:.4f}")
                m_va = bin_metrics_from_probs(p_fuse_va_np, y_val_arr, tau)
                print(f"  -> prec={m_va['precision']:.4f} rec={m_va['recall']:.4f}")

                if f1_ > best_f1:
                    best_f1, best_tau = f1_, tau
                    best_state = {k: v.detach().cpu() for k, v in gate.state_dict().items()}

    # 保存最佳 Gate
    if best_state is not None:
        gate.load_state_dict(best_state, strict=True)

    gate.eval()
    Zt = torch.tensor(phi_te_n, dtype=torch.float32, device=device)
    gt = gate(Zt)

    y_test_arr = np.load(os.path.join(root, "y_test.npy"))

    pr_te_t = torch.tensor(pr_te, dtype=torch.float32, device=device)
    pc_te_t = torch.tensor(pc_te, dtype=torch.float32, device=device)
    p_fuse_te = torch.sigmoid(gt * logit_t(pr_te_t) + (1.0 - gt) * logit_t(pc_te_t)).detach().cpu().numpy()
    m_te = bin_metrics_from_probs(p_fuse_te, y_test_arr, best_tau)

    print(f"\n[FUSE] TEST @ tau={best_tau:.2f} | acc={m_te['acc']:.4f}  f1={m_te['f1']:.4f}  "
        f"prec={m_te['precision']:.4f} rec={m_te['recall']:.4f}")
    print("[FUSE] Confusion Matrix [[TN,FP],[FN,TP]]:\n", m_te["cm"])

    base_r = bin_metrics_from_probs(pr_te, y_test_arr, 0.5)
    base_c = bin_metrics_from_probs(pc_te, y_test_arr, 0.5)
    print(f"\n[BASE] Darknet(2cls) TEST@{0.5:.2f} acc={base_r['acc']:.4f} f1={base_r['f1']:.4f}")
    print(f"[BASE] Hier(4cls->bin) TEST@{0.5:.2f} acc={base_c['acc']:.4f} f1={base_c['f1']:.4f}")

    local_ckpt_dir = cfg.CKPT_DIR
    local_log_dir  = cfg.LOG_DIR
    os.makedirs(local_ckpt_dir, exist_ok=True)
    os.makedirs(local_log_dir,  exist_ok=True)

    ckpt_path = os.path.join(local_ckpt_dir, f"cdgf_gatenet_best_{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt")
    torch.save({
        "model": "GateNet_CDGF_darknet",
        "state_dict": {k: v.cpu() for k, v in gate.state_dict().items()},
        "phi_mu": mu_phi, "phi_std": std_phi,
        "best_tau": best_tau,
        "ckpt_2cls": args.ckpt_2cls,
        "ckpt_4cls": args.ckpt_4cls,
        "map_pos_idx": map_pos_idx
    }, ckpt_path)
    print(f"[CKPT] gate saved: {ckpt_path}")

    log_path = os.path.join(local_log_dir, f"metrics_cdgf_darknet_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tau","fuse_acc","fuse_f1","fuse_prec","fuse_rec","fuse_bal_acc"])
        w.writerow([f"{best_tau:.4f}", f"{m_te['acc']:.6f}", f"{m_te['f1']:.6f}",
                    f"{m_te['precision']:.6f}", f"{m_te['recall']:.6f}", f"{m_te['bal_acc']:.6f}"])
        w.writerow([])
        w.writerow(["darknet2_tau","darknet2_acc","darknet2_f1"])
        w.writerow([f"{0.5:.4f}", f"{base_r['acc']:.6f}", f"{base_r['f1']:.6f}"])
        w.writerow(["hier4_tau","hier4_acc","hier4_f1"])
        w.writerow([f"{0.5:.4f}", f"{base_c['acc']:.6f}", f"{base_c['f1']:.6f}"])
    print(f"[LOG] saved: {log_path}")

if __name__ == "__main__":
    main()
