import os, csv, argparse, time
import numpy as np
import librosa
from joblib import dump
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg


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
):
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
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0).astype(np.float32)  # [2*n_mfcc]
    return feat


def extract_mfcc_stats_split(X: np.ndarray, args, tag: str):
    feats = []
    for i in range(X.shape[0]):
        f = mfcc_stats_1d(
            X[i],
            sr=args.sr,
            n_fft=args.n_fft,
            win_length=args.win_length,
            hop_length=args.hop_length,
            window=args.window,
            n_mels=args.n_mels,
            n_mfcc=args.n_mfcc,
            fmin=args.fmin,
            fmax=args.fmax,
            center=bool(args.center),
        )
        feats.append(f)
        if (i + 1) % 200 == 0 or (i + 1) == X.shape[0]:
            print(f"[MFCC-{tag}] {i+1}/{X.shape[0]} done")
    return np.vstack(feats).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET)

    # MFCC 参数（按你的场景做工程化默认）
    ap.add_argument("--sr", type=int, default=getattr(cfg, "FS", 8192))
    ap.add_argument("--n_fft", type=int, default=256)
    ap.add_argument("--win_length", type=int, default=256)
    ap.add_argument("--hop_length", type=int, default=128)
    ap.add_argument("--window", type=str, default="hamming")
    ap.add_argument("--n_mels", type=int, default=40)
    ap.add_argument("--n_mfcc", type=int, default=20)
    ap.add_argument("--fmin", type=float, default=20.0)
    ap.add_argument("--fmax", type=float, default=-1.0)  # <=0 时自动取 sr/2
    ap.add_argument("--center", type=int, default=1)     # 1=True, 0=False

    ap.add_argument("--c_grid", type=str, default="0.1,0.3,1.0,3.0,10.0")
    ap.add_argument("--thr_lo", type=float, default=0.20)
    ap.add_argument("--thr_hi", type=float, default=0.80)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    if args.fmax <= 0:
        args.fmax = float(args.sr) / 2.0
    if args.fmin < 0 or args.fmin >= args.fmax:
        raise ValueError(f"Invalid fmin/fmax: fmin={args.fmin}, fmax={args.fmax}")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- 读取原始 1D 信号与二分类标签 ----
    Xtr = np.load(os.path.join(args.data_root, "X_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "X_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "X_test.npy"))
    ytr = np.load(os.path.join(args.data_root, "y_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, "y_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, "y_test.npy")).astype(np.int64)

    print("Extracting MFCC mean+std features ...")
    Ftr = extract_mfcc_stats_split(Xtr, args, tag="train")
    Fva = extract_mfcc_stats_split(Xva, args, tag="val")
    Fte = extract_mfcc_stats_split(Xte, args, tag="test")

    # 用 train 统计做标准化
    mu = Ftr.mean(axis=0, keepdims=True)
    sg = Ftr.std(axis=0, keepdims=True) + 1e-6
    Ftr = (Ftr - mu) / sg
    Fva = (Fva - mu) / sg
    Fte = (Fte - mu) / sg

    # ---- 训练与验证（扫描 C 与阈值）----
    C_GRID = [float(s) for s in args.c_grid.split(",") if s.strip()]
    best_model, best_C, best_thr, best_val_m = None, None, 0.5, None
    pos_index = None
    for C in C_GRID:
        clf = SVC(kernel="linear", C=C, class_weight="balanced", probability=True, random_state=args.seed)
        clf.fit(Ftr, ytr)

        proba_va = clf.predict_proba(Fva)  # [N,2], 列顺序与 clf.classes_ 对齐
        if pos_index is None:
            pos_index = int(np.where(clf.classes_ == 1)[0][0])
        p_va = proba_va[:, pos_index]

        thr, f1_opt, m_opt = find_best_threshold(p_va, yva, lo=args.thr_lo, hi=args.thr_hi, step=args.thr_step)
        print(
            f"C={C:<5} | val@thr={thr:.2f} acc={m_opt['acc']:.4f} f1={m_opt['f1']:.4f} "
            f"prec={m_opt['precision']:.4f} rec={m_opt['recall']:.4f}"
        )

        if (best_val_m is None) or (m_opt["f1"] > best_val_m["f1"]) or \
           (abs(m_opt["f1"] - best_val_m["f1"]) < 1e-12 and m_opt["acc"] > best_val_m["acc"]):
            best_model, best_C, best_thr, best_val_m = clf, C, thr, m_opt

    # ---- 验证总结 ----
    print(
        f"\n[VAL-BEST] C={best_C} thr={best_thr:.2f} | "
        f"acc={best_val_m['acc']:.4f} f1={best_val_m['f1']:.4f} "
        f"prec={best_val_m['precision']:.4f} rec={best_val_m['recall']:.4f}"
    )

    # ---- 测试评估（沿用同一阈值）----
    proba_te = best_model.predict_proba(Fte)[:, pos_index]
    m_te = bin_metrics_from_probs(proba_te, yte, best_thr)
    print(
        f"\n[TEST] thr={best_thr:.2f} | acc={m_te['acc']:.4f} f1={m_te['f1']:.4f} "
        f"prec={m_te['precision']:.4f} rec={m_te['recall']:.4f}"
    )

    # ---- 混淆矩阵 ----
    yhat_va = (best_model.predict_proba(Fva)[:, pos_index] >= best_thr).astype(np.int64)
    yhat_te = (proba_te >= best_thr).astype(np.int64)
    cm_val = confusion_matrix(yva, yhat_va, labels=[0, 1])
    cm_te = confusion_matrix(yte, yhat_te, labels=[0, 1])
    print(f"\n[VAL]  Confusion Matrix [[TN,FP],[FN,TP]]:\n{cm_val}")
    print(f"\n[TEST] Confusion Matrix [[TN,FP],[FN,TP]]:\n{cm_te}")

    # ---- 保存模型与日志 ----
    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(args.save_dir, f"svm_mfcc_2cls_best_{stamp}.joblib")
    dump(
        {
            "model": best_model,
            "best_thr": best_thr,
            "C": best_C,
            "pos_index": pos_index,
            "mfcc_cfg": {
                "sr": args.sr,
                "n_fft": args.n_fft,
                "win_length": args.win_length,
                "hop_length": args.hop_length,
                "window": args.window,
                "n_mels": args.n_mels,
                "n_mfcc": args.n_mfcc,
                "fmin": args.fmin,
                "fmax": args.fmax,
                "center": bool(args.center),
            },
            "feat_mu": mu.squeeze(0).astype(np.float32),
            "feat_sigma": sg.squeeze(0).astype(np.float32),
        },
        ckpt_path,
    )

    log_path = os.path.join(args.log_dir, f"metrics_svm_mfcc_2cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "C", "val_acc", "val_f1", "val_precision", "val_recall",
                "best_thr", "test_acc", "test_f1", "test_precision", "test_recall",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "C": best_C,
                "val_acc": best_val_m["acc"],
                "val_f1": best_val_m["f1"],
                "val_precision": best_val_m["precision"],
                "val_recall": best_val_m["recall"],
                "best_thr": best_thr,
                "test_acc": m_te["acc"],
                "test_f1": m_te["f1"],
                "test_precision": m_te["precision"],
                "test_recall": m_te["recall"],
            }
        )
    print(f"\n[CKPT] {ckpt_path}\n[LOG]  {log_path}")


if __name__ == "__main__":
    main()
