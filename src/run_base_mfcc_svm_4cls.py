import os, csv, argparse, time
import numpy as np
import librosa
from joblib import dump
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg


def mc_metrics_from_pred(y_pred: np.ndarray, y_true: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


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
    feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)], axis=0).astype(np.float32)
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
    ap.add_argument("--label_prefix", type=str, default="y4")

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

    # SVM 参数
    ap.add_argument("--c_grid", type=str, default="0.1,0.3,1.0,3.0,10.0")
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

    # 读取原始 1D 信号与四分类标签
    Xtr = np.load(os.path.join(args.data_root, "X_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "X_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "X_test.npy"))
    ytr = np.load(os.path.join(args.data_root, f"{args.label_prefix}_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, f"{args.label_prefix}_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, f"{args.label_prefix}_test.npy")).astype(np.int64)

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

    C_GRID = [float(s) for s in args.c_grid.split(",") if s.strip()]
    if not C_GRID:
        raise ValueError("Empty --c_grid")

    best_model, best_C, best_val_m = None, None, None
    for C in C_GRID:
        clf = SVC(
            kernel="linear",
            C=C,
            class_weight="balanced",
            probability=False,
            decision_function_shape="ovr",
            random_state=args.seed,
        )
        clf.fit(Ftr, ytr)

        yhat_va = clf.predict(Fva)
        m_va = mc_metrics_from_pred(yhat_va, yva)
        print(
            f"C={C:<5} | val acc={m_va['acc']:.4f} macro-F1={m_va['f1']:.4f} "
            f"P_macro={m_va['precision']:.4f} R_macro={m_va['recall']:.4f}"
        )

        if (best_val_m is None) or (m_va["f1"] > best_val_m["f1"]) or \
           (abs(m_va["f1"] - best_val_m["f1"]) < 1e-12 and m_va["acc"] > best_val_m["acc"]):
            best_model, best_C, best_val_m = clf, C, m_va

    print(
        f"\n[VAL-BEST] C={best_C} | acc={best_val_m['acc']:.4f} macro-F1={best_val_m['f1']:.4f} "
        f"P_macro={best_val_m['precision']:.4f} R_macro={best_val_m['recall']:.4f}"
    )

    yhat_te = best_model.predict(Fte)
    m_te = mc_metrics_from_pred(yhat_te, yte)
    print(
        f"\n[TEST]  acc={m_te['acc']:.4f} macro-F1={m_te['f1']:.4f} "
        f"P_macro={m_te['precision']:.4f} R_macro={m_te['recall']:.4f}"
    )

    num_classes = int(max(np.max(ytr), np.max(yva), np.max(yte))) + 1
    labels = list(range(num_classes))
    cm_val = confusion_matrix(yva, best_model.predict(Fva), labels=labels)
    cm_te = confusion_matrix(yte, yhat_te, labels=labels)
    print(f"\n[VAL]  Confusion Matrix (rows=true, cols=pred):\n{cm_val}")
    print(f"\n[TEST] Confusion Matrix (rows=true, cols=pred):\n{cm_te}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(args.save_dir, f"svm_mfcc_4cls_best_{stamp}.joblib")
    dump(
        {
            "model": best_model,
            "C": best_C,
            "classes_": getattr(best_model, "classes_", None),
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

    log_path = os.path.join(args.log_dir, f"metrics_svm_mfcc_4cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "C", "val_acc", "val_macro_f1", "val_precision_macro", "val_recall_macro",
                "test_acc", "test_macro_f1", "test_precision_macro", "test_recall_macro",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "C": best_C,
                "val_acc": best_val_m["acc"],
                "val_macro_f1": best_val_m["f1"],
                "val_precision_macro": best_val_m["precision"],
                "val_recall_macro": best_val_m["recall"],
                "test_acc": m_te["acc"],
                "test_macro_f1": m_te["f1"],
                "test_precision_macro": m_te["precision"],
                "test_recall_macro": m_te["recall"],
            }
        )

    print(f"\n[CKPT] {ckpt_path}\n[LOG]  {log_path}")


if __name__ == "__main__":
    main()
