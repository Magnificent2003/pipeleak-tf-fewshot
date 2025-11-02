import os, shutil
import numpy as np
import pywt

import config as cfg

# ===== 计算尺度（按对数频率均匀取样）=====
def make_scales(fs, wavelet, n_scales, fmin, fmax):
    freqs = np.geomspace(fmin, fmax, num=n_scales)
    dt = 1.0 / fs
    cf = pywt.central_frequency(wavelet)
    scales = cf / (freqs * dt)
    return scales.astype(np.float64), freqs.astype(np.float64)

# ===== 单条样本 -> 特征向量 =====
def cwt_features_1d(sig, fs, wavelet, scales, use_log=True):
    x = sig - np.mean(sig)
    coefs, _freqs = pywt.cwt(x, scales, wavelet, sampling_period=1.0/fs)
    mag = np.abs(coefs).astype(np.float32)  # [S, T]
    if use_log:
        mag = np.log1p(mag)
    m  = mag.mean(axis=1)
    sd = mag.std (axis=1)
    q1 = np.quantile(mag, 0.25, axis=1)
    q3 = np.quantile(mag, 0.75, axis=1)
    feat = np.stack([m, sd, q1, q3], axis=1).astype(np.float32)  # [S, 4]
    return feat.reshape(-1)  # [S*4]

def process_split(X, y, fs, wavelet, scales, use_log=True, tag="train"):
    feats = []
    for i in range(X.shape[0]):
        f = cwt_features_1d(X[i], fs, wavelet, scales, use_log=use_log)
        feats.append(f)
        if (i+1) % 100 == 0:
            print(f"[{tag}] {i+1}/{X.shape[0]} done")
    F = np.vstack(feats).astype(np.float32)  # [N, S*4]
    return F, y.astype(np.int64)

def main():
    root = cfg.DATASET
    save_root = cfg.DATASET_CWT
    os.makedirs(save_root, exist_ok=True)
    X_tr = np.load(os.path.join(root, "X_train.npy"))
    X_va = np.load(os.path.join(root, "X_val.npy"))
    X_te = np.load(os.path.join(root, "X_test.npy"))
    y_tr = np.load(os.path.join(root, "y_train.npy")).astype(np.int64)
    y_va = np.load(os.path.join(root, "y_val.npy")).astype(np.int64)
    y_te = np.load(os.path.join(root, "y_test.npy")).astype(np.int64)

    print(f"Loaded: train={X_tr.shape}, val={X_va.shape}, test={X_te.shape}")

    scales, freqs = make_scales(cfg.FS, cfg.WAVELET, cfg.N_SCALES, cfg.FMIN_HZ, cfg.FMAX_HZ)
    print(f"CWT scales={len(scales)} | f[{freqs.min():.1f}..{freqs.max():.1f}] Hz")

    Xc_tr, yc_tr = process_split(X_tr, y_tr, cfg.FS, cfg.WAVELET, scales, cfg.USE_LOG1P, tag="train")
    Xc_va, yc_va = process_split(X_va, y_va, cfg.FS, cfg.WAVELET, scales, cfg.USE_LOG1P, tag="val")
    Xc_te, yc_te = process_split(X_te, y_te, cfg.FS, cfg.WAVELET, scales, cfg.USE_LOG1P, tag="test")

    print("Raw CWT feature shapes:", Xc_tr.shape, Xc_va.shape, Xc_te.shape)

    # ===== 用 train 的逐维统计做标准化（更合适于手工特征）=====
    mu = Xc_tr.mean(axis=0, keepdims=True)
    sg = Xc_tr.std (axis=0, keepdims=True) + cfg.EPS
    Xc_tr = (Xc_tr - mu) / sg
    Xc_va = (Xc_va - mu) / sg
    Xc_te = (Xc_te - mu) / sg

    # ===== 保存 =====
    np.save(os.path.join(save_root, "Xcwt_train.npy"), Xc_tr.astype(np.float32))
    np.save(os.path.join(save_root, "Xcwt_val.npy"),   Xc_va.astype(np.float32))
    np.save(os.path.join(save_root, "Xcwt_test.npy"),  Xc_te.astype(np.float32))
    np.save(os.path.join(save_root, "y_train.npy"), yc_tr)
    np.save(os.path.join(save_root, "y_val.npy"),   yc_va)
    np.save(os.path.join(save_root, "y_test.npy"),  yc_te)
    shutil.copy(os.path.join(root, "y4_train.npy"), os.path.join(save_root, "y4_train.npy")) 
    shutil.copy(os.path.join(root, "y4_val.npy"),   os.path.join(save_root, "y4_val.npy")) 
    shutil.copy(os.path.join(root, "y4_test.npy"),  os.path.join(save_root, "y4_test.npy"))

    np.save(os.path.join(save_root, "Xcwt_mu.npy"),    mu.astype(np.float32).squeeze(0))
    np.save(os.path.join(save_root, "Xcwt_sigma.npy"), sg.astype(np.float32).squeeze(0))
    np.save(os.path.join(save_root, "cwt_freqs.npy"),  freqs.astype(np.float32))

    print("Saved:")
    print("  Xcwt_[train/val/test].npy  (features, shape ~ [N, 4*S])")
    print("  ycwt_[train/val/test].npy  (labels)")
    print("  Xcwt_mu.npy / Xcwt_sigma.npy (per-dim standardization)")
    print("  cwt_freqs.npy (Hz grid)")
    print("Done.")

if __name__ == "__main__":
    main()
