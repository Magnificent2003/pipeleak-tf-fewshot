import os, csv, argparse, time
import numpy as np
from scipy.signal import hilbert
from joblib import dump
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

import config as cfg

try:
    from PyEMD import CEEMDAN, EEMD
except Exception as e:
    CEEMDAN = None
    EEMD = None
    _PYEMD_IMPORT_ERR = e


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


def _build_hht_cache_key(args):
    key = (
        f"m{args.hht_method}_sr{args.sr}_tr{args.hht_trials}_nw{args.hht_noise_width}_"
        f"max{args.hht_max_imf}_keep{args.hht_keep_imf}_"
        f"fb{args.hht_freq_bins}_fmin{args.hht_fmin}_fmax{args.hht_fmax}"
    )
    return key.replace(".", "p")


def _make_decomposer(args):
    if args.hht_method == "ceemdan":
        if CEEMDAN is None:
            raise ImportError(
                f"PyEMD/CEEMDAN is required for HHT baseline. Original error: {_PYEMD_IMPORT_ERR}"
            )
        dec = CEEMDAN(trials=args.hht_trials, parallel=False)
        if hasattr(dec, "epsilon"):
            dec.epsilon = float(args.hht_noise_width)
        if hasattr(dec, "noise_scale"):
            dec.noise_scale = 1.0
        return dec

    if args.hht_method == "eemd":
        if EEMD is None:
            raise ImportError(
                f"PyEMD/EEMD is required for HHT baseline. Original error: {_PYEMD_IMPORT_ERR}"
            )
        dec = EEMD(trials=args.hht_trials, parallel=False)
        if hasattr(dec, "noise_width"):
            dec.noise_width = float(args.hht_noise_width)
        return dec

    raise ValueError(f"Unsupported hht_method: {args.hht_method}")


def hht_feature_1d(sig: np.ndarray, decomposer, args, bin_edges: np.ndarray):
    x = np.asarray(sig, dtype=np.float64).reshape(-1)
    x = x - np.mean(x)
    if np.std(x) < 1e-12:
        return np.zeros(args.hht_freq_bins + args.hht_keep_imf, dtype=np.float32)

    try:
        if args.hht_method == "ceemdan":
            try:
                imfs = decomposer.ceemdan(x, max_imf=args.hht_max_imf, progress=False)
            except TypeError:
                imfs = decomposer.ceemdan(x, max_imf=args.hht_max_imf)
        else:
            try:
                imfs = decomposer.eemd(x, max_imf=args.hht_max_imf, progress=False)
            except TypeError:
                imfs = decomposer.eemd(x, max_imf=args.hht_max_imf)
    except Exception:
        imfs = x[None, :]

    imfs = np.asarray(imfs, dtype=np.float64)
    if imfs.ndim == 1:
        imfs = imfs[None, :]
    if imfs.shape[0] == 0:
        imfs = x[None, :]

    # 仅保留前 keep_imf 个 IMF，不纳入 residual
    imfs = imfs[: args.hht_keep_imf]

    marginal = np.zeros(args.hht_freq_bins, dtype=np.float64)
    imf_energy = np.zeros(args.hht_keep_imf, dtype=np.float64)

    for k in range(imfs.shape[0]):
        imf = imfs[k]
        if np.allclose(imf, 0):
            continue

        z = hilbert(imf)
        amp = np.abs(z)
        phase = np.unwrap(np.angle(z))

        inst_freq = np.diff(phase) * (args.sr / (2.0 * np.pi))
        weights = (amp[:-1] ** 2)

        valid = (
            np.isfinite(inst_freq)
            & np.isfinite(weights)
            & (inst_freq >= args.hht_fmin)
            & (inst_freq <= args.hht_fmax)
        )
        if np.any(valid):
            h, _ = np.histogram(inst_freq[valid], bins=bin_edges, weights=weights[valid])
            marginal += h

        imf_energy[k] = float(np.sum(imf ** 2))

    ms = float(np.sum(marginal))
    if ms > 0:
        marginal = marginal / (ms + 1e-12)

    es = float(np.sum(imf_energy))
    if es > 0:
        imf_ratio = imf_energy / (es + 1e-12)
    else:
        imf_ratio = np.zeros_like(imf_energy)

    feat = np.concatenate([marginal.astype(np.float32), imf_ratio.astype(np.float32)], axis=0)
    return feat


def extract_hht_split(X: np.ndarray, args, tag: str, seed_offset: int = 0):
    decomposer = _make_decomposer(args)
    bin_edges = np.linspace(args.hht_fmin, args.hht_fmax, args.hht_freq_bins + 1, dtype=np.float64)
    feats = []
    for i in range(X.shape[0]):
        if hasattr(decomposer, "noise_seed"):
            decomposer.noise_seed(int(args.seed + seed_offset + i))
        f = hht_feature_1d(X[i], decomposer, args, bin_edges)
        feats.append(f)
        if (i + 1) % 50 == 0 or (i + 1) == X.shape[0]:
            print(f"[HHT-{args.hht_method}-{tag}] {i+1}/{X.shape[0]} done")
    return np.vstack(feats).astype(np.float32)


def load_or_build_hht_features(Xtr, Xva, Xte, args):
    cache_dir = args.hht_cache_dir
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        key = _build_hht_cache_key(args)
        p_tr = os.path.join(cache_dir, f"Xhht_train_{key}.npy")
        p_va = os.path.join(cache_dir, f"Xhht_val_{key}.npy")
        p_te = os.path.join(cache_dir, f"Xhht_test_{key}.npy")
        if (not args.hht_force_rebuild) and all(os.path.exists(p) for p in [p_tr, p_va, p_te]):
            print(f"[HHT] Load cached features from {cache_dir}")
            return np.load(p_tr), np.load(p_va), np.load(p_te)

    print(f"[HHT] Extracting {args.hht_method.upper()} + Hilbert marginal spectrum features ...")
    Ftr = extract_hht_split(Xtr, args, tag="train", seed_offset=0)
    Fva = extract_hht_split(Xva, args, tag="val", seed_offset=100000)
    Fte = extract_hht_split(Xte, args, tag="test", seed_offset=200000)

    if cache_dir:
        key = _build_hht_cache_key(args)
        p_tr = os.path.join(cache_dir, f"Xhht_train_{key}.npy")
        p_va = os.path.join(cache_dir, f"Xhht_val_{key}.npy")
        p_te = os.path.join(cache_dir, f"Xhht_test_{key}.npy")
        np.save(p_tr, Ftr)
        np.save(p_va, Fva)
        np.save(p_te, Fte)
        print(f"[HHT] Saved cached features to {cache_dir}")

    return Ftr, Fva, Fte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET)

    # HHT 参数（推荐配置）
    ap.add_argument("--sr", type=int, default=getattr(cfg, "FS", 8192))
    ap.add_argument("--hht_method", type=str, default="eemd", choices=["eemd", "ceemdan"])
    ap.add_argument("--hht_trials", type=int, default=20)
    ap.add_argument("--hht_noise_width", type=float, default=0.2)
    ap.add_argument("--hht_max_imf", type=int, default=6)
    ap.add_argument("--hht_keep_imf", type=int, default=5)
    ap.add_argument("--hht_freq_bins", type=int, default=64)
    ap.add_argument("--hht_fmin", type=float, default=0.0)
    ap.add_argument("--hht_fmax", type=float, default=-1.0)  # <=0 时自动取 sr/2

    # 可选缓存（HHT 提取较慢，默认开启）
    ap.add_argument("--hht_cache_dir", type=str, default=os.path.join(cfg.DATA_DIR, "dataset_hht"))
    ap.add_argument("--hht_force_rebuild", type=int, default=0)

    # SVM 参数
    ap.add_argument("--c_grid", type=str, default="0.1,0.3,1.0,3.0,10.0")
    ap.add_argument("--thr_lo", type=float, default=0.20)
    ap.add_argument("--thr_hi", type=float, default=0.80)
    ap.add_argument("--thr_step", type=float, default=0.01)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args()

    if args.hht_fmax <= 0:
        args.hht_fmax = float(args.sr) / 2.0
    if args.hht_fmin < 0 or args.hht_fmin >= args.hht_fmax:
        raise ValueError(f"Invalid hht_fmin/hht_fmax: {args.hht_fmin}, {args.hht_fmax}")

    if args.hht_keep_imf <= 0 or args.hht_max_imf <= 0 or args.hht_keep_imf > args.hht_max_imf:
        raise ValueError("Require: 0 < hht_keep_imf <= hht_max_imf")

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- 读取原始 1D 信号与二分类标签 ----
    Xtr = np.load(os.path.join(args.data_root, "X_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "X_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "X_test.npy"))
    ytr = np.load(os.path.join(args.data_root, "y_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, "y_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, "y_test.npy")).astype(np.int64)

    Ftr, Fva, Fte = load_or_build_hht_features(Xtr, Xva, Xte, args)

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
    ckpt_path = os.path.join(args.save_dir, f"svm_hht_2cls_best_{stamp}.joblib")
    dump(
        {
            "model": best_model,
            "best_thr": best_thr,
            "C": best_C,
            "pos_index": pos_index,
            "hht_cfg": {
                "sr": args.sr,
                "hht_method": args.hht_method,
                "hht_trials": args.hht_trials,
                "hht_noise_width": args.hht_noise_width,
                "hht_max_imf": args.hht_max_imf,
                "hht_keep_imf": args.hht_keep_imf,
                "hht_freq_bins": args.hht_freq_bins,
                "hht_fmin": args.hht_fmin,
                "hht_fmax": args.hht_fmax,
            },
            "feat_mu": mu.squeeze(0).astype(np.float32),
            "feat_sigma": sg.squeeze(0).astype(np.float32),
        },
        ckpt_path,
    )
    log_path = os.path.join(args.log_dir, f"metrics_svm_hht_2cls_{stamp}.csv")
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
