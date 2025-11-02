# eval_fuse_network.py (minimal)
# -*- coding: utf-8 -*-
import os, sys, argparse, numpy as np, torch
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.config as cfg
from src.NpyDataset import NpyDataset
from src.Darknet19 import Darknet19
from src.HierarchicalDarknet19 import HierarchicalDarknet19
from src.Gatenet import Gatenet

def bin_metrics_from_probs(p, y, thr):
    y_hat = (p >= thr).astype(np.int64)
    cm = confusion_matrix(y, y_hat, labels=[0,1])
    acc = accuracy_score(y, y_hat)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_hat, average="binary", zero_division=0)
    bal_acc = 0.5 * (cm[0,0]/max(cm[0,0]+cm[0,1],1) + cm[1,1]/max(cm[1,1]+cm[1,0],1))
    return acc, f1, prec, rec, bal_acc, cm

def entropy01_np(p, eps=1e-9):
    p = np.clip(p, eps, 1.0-eps)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def make_phi(pr, pc):
    return np.stack([pr, pc, np.abs(pr-pc), entropy01_np(pr), entropy01_np(pc)], axis=1).astype(np.float32)

@torch.no_grad()
def probs_2cls(model, loader, device, eps=1e-9):
    model.eval(); out = []
    for x, _ in loader:
        p = torch.softmax(model(x.to(device)), dim=1)[:,1].cpu().numpy()
        out.append(np.clip(p, eps, 1-eps))
    return np.concatenate(out, 0)

@torch.no_grad()
def probs_4cls(model, loader, pos_idx, device, eps=1e-9):
    model.eval(); out = []
    for x, _ in loader:
        o = model(x.to(device))
        logits = o[1] if (isinstance(o,(tuple,list)) and len(o)>=2) else o
        p = torch.softmax(logits, dim=1)[:, pos_idx].sum(dim=1).cpu().numpy()
        out.append(np.clip(p, eps, 1-eps))
    return np.concatenate(out, 0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--ckpt_gate", type=str, default="./checkpoints/cdgf_gatenet_best.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    Xte = os.path.join(args.data_root, f"X_test_stft_{args.img_size}.npy")
    yte = os.path.join(args.data_root, "y_test.npy")
    ds_te = NpyDataset(Xte, yte, normalize="imagenet", memmap=True)
    te_loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)
    y_test = np.load(yte).astype(np.int64)

    # Gate ckpt（必须包含所需元信息）
    gckpt = torch.load(args.ckpt_gate, map_location="cpu", weights_only=False)
    tau  = float(gckpt["best_tau"])
    mu   = np.asarray(gckpt["phi_mu"],  dtype=np.float32).reshape(1, -1)
    std  = np.asarray(gckpt["phi_std"], dtype=np.float32).reshape(1, -1)
    ck2  = gckpt["ckpt_2cls"]
    ck4  = gckpt["ckpt_4cls"]
    pos_idx = gckpt.get("map_pos_idx", [0,1])
    tau_r = float(gckpt.get("tau_r", 0.5))

    # 2cls
    c2 = torch.load(ck2, map_location="cpu", weights_only=False)
    m2 = Darknet19(num_classes=int(c2.get("num_classes",2))).to(device).eval()
    s2 = c2.get("state_dict", c2); m2.load_state_dict({k[7:] if k.startswith("module.") else k:v for k,v in s2.items()}, strict=False)

    # 4cls（层级）
    c4 = torch.load(ck4, map_location="cpu", weights_only=False)
    m4 = HierarchicalDarknet19(num_classes=int(c4.get("num_classes",4))).to(device).eval()
    s4 = c4.get("state_dict", c4); m4.load_state_dict({k[7:] if k.startswith("module.") else k:v for k,v in s4.items()}, strict=False)

    # Gate
    gate = Gatenet(in_dim=5, hid=int(gckpt.get("hid",16))).to(device).eval()
    gate.load_state_dict({k: (v.to(device) if hasattr(v,"to") else torch.tensor(v).to(device))
                          for k,v in gckpt["state_dict"].items()}, strict=True)

    # 推理
    pr_te = probs_2cls(m2, te_loader, device)
    pc_te = probs_4cls(m4, te_loader, pos_idx, device)
    phi_te = make_phi(pr_te, pc_te)
    phi_te_n = (phi_te - mu) / (std + 1e-6)

    with torch.no_grad():
        g = gate(torch.tensor(phi_te_n, dtype=torch.float32, device=device)).cpu().numpy().squeeze()
    p = np.clip(g*pr_te + (1-g)*pc_te, 1e-9, 1-1e-9)

    # 评估（仅用 ckpt 中的 tau）
    acc, f1, prec, rec, bal_acc, cm = bin_metrics_from_probs(p, y_test, tau)
    print("=== Fuse (Gate ckpt, fixed tau) ===")
    print(f"tau={tau:.4f} | acc={acc:.6f} | f1={f1:.6f} | prec={prec:.6f} | rec={rec:.6f} | bal_acc={bal_acc:.6f}")
    print("Confusion Matrix [[TN,FP],[FN,TP]]:\n", cm)

if __name__ == "__main__":
    main()
