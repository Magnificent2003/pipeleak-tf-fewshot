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


def mc_metrics_from_pred(y_pred: np.ndarray, y_true: np.ndarray):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


def parent_metrics_from_cm4(cm4: np.ndarray):
    # Parent mapping: {0,1}->0 and {2,3}->1
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
    ap.add_argument("--label_prefix", type=str, default="y4")

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

    # 读取原始 1D 信号与四分类标签
    Xtr = np.load(os.path.join(args.data_root, "X_train.npy"))
    Xva = np.load(os.path.join(args.data_root, "X_val.npy"))
    Xte = np.load(os.path.join(args.data_root, "X_test.npy"))
    ytr = np.load(os.path.join(args.data_root, f"{args.label_prefix}_train.npy")).astype(np.int64)
    yva = np.load(os.path.join(args.data_root, f"{args.label_prefix}_val.npy")).astype(np.int64)
    yte = np.load(os.path.join(args.data_root, f"{args.label_prefix}_test.npy")).astype(np.int64)

    Ftr, Fva, Fte = load_or_build_hht_features(Xtr, Xva, Xte, args)

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
    cm_parent_te, parent_f1_te, parent_rec_te = parent_metrics_from_cm4(cm_te)
    print(f"\n[VAL]  Confusion Matrix (rows=true, cols=pred):\n{cm_val}")
    print(f"\n[TEST] Confusion Matrix (rows=true, cols=pred):\n{cm_te}")
    print(f"\n[TEST] Parent Confusion Matrix (rows=true, cols=pred, 01->0,23->1):\n{cm_parent_te}")
    print(f"[TEST] 4-class Parent-F1={parent_f1_te:.4f} Parent-Recall={parent_rec_te:.4f}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    ckpt_path = os.path.join(args.save_dir, f"svm_hht_4cls_best_{stamp}.joblib")
    dump(
        {
            "model": best_model,
            "C": best_C,
            "classes_": getattr(best_model, "classes_", None),
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

    log_path = os.path.join(args.log_dir, f"metrics_svm_hht_4cls_{stamp}.csv")
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "C", "val_acc", "val_macro_f1", "val_precision_macro", "val_recall_macro",
                "test_acc", "test_macro_f1", "test_precision_macro", "test_recall_macro",
                "test_parent_f1", "test_parent_recall",
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
                "test_parent_f1": parent_f1_te,
                "test_parent_recall": parent_rec_te,
            }
        )

    print(f"\n[CKPT] {ckpt_path}\n[LOG]  {log_path}")


if __name__ == "__main__":
    main()
