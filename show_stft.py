import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

import src.config as cfg  # 和你 build_stft_datasets.py 保持一致

import matplotlib as mpl
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"   # 数学符号更接近 Times 风格
mpl.rcParams["axes.unicode_minus"] = False  # 负号正常显示

mpl.rcParams["font.size"] = 12
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["axes.labelsize"] = 12
mpl.rcParams["xtick.labelsize"] = 11
mpl.rcParams["ytick.labelsize"] = 11


# ========= 在这里指定你要看的样本 =========
IDX = 0
SPLIT = "train"  # "train"/"val"/"test"
CMAP = "turbo"   # 你也可以改成 "jet" / "viridis" / "magma"

# ========= 读取数据 =========
X_path = os.path.join(cfg.DATASET, f"X_{SPLIT}.npy")
y4_path = os.path.join(cfg.DATASET, f"y4_{SPLIT}.npy")

X = np.load(X_path, mmap_mode="r")
y4 = np.load(y4_path, mmap_mode="r")

x = np.asarray(X[IDX]).reshape(-1).astype(np.float32)
y4_val = int(np.asarray(y4[IDX]).reshape(-1)[0])

print(f"[{SPLIT}] idx={IDX}, y4={y4_val}")

# ========= 复刻你当时的 STFT 灰度谱图生成逻辑（只是不保存灰度，直接彩色显示）=========
D = librosa.stft(
    x,
    n_fft=cfg.N_FFT,
    hop_length=cfg.HOP_LENGTH,
    window=cfg.WINDOW,
    center=cfg.CENTER,
)
S = np.abs(D)
S_db = librosa.amplitude_to_db(S, ref=np.max)
S_db = np.clip(S_db, cfg.DB_MIN, cfg.DB_MAX)
spec01 = (S_db - cfg.DB_MIN) / (cfg.DB_MAX - cfg.DB_MIN)  # [0,1]

# ========= 坐标轴：时间 & 频率 =========
fs = float(getattr(cfg, "FS", 1.0))  # 若 config 没 FS，就用 1.0（单位会变成“frame”意义）
freqs = librosa.fft_frequencies(sr=fs, n_fft=cfg.N_FFT)                 # Hz
times = librosa.frames_to_time(np.arange(spec01.shape[1]), sr=fs, hop_length=cfg.HOP_LENGTH)  # s

extent = [times.min(), times.max(), freqs.min(), freqs.max()]

# ========= 显示（彩色 + 坐标轴 + colorbar）=========
plt.figure(figsize=(7, 4.5))
im = plt.imshow(
    spec01,
    origin="lower",
    aspect="auto",
    cmap=CMAP,
    vmin=0.0,
    vmax=1.0,
    extent=extent,
)
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
# plt.title(f"STFT Spectrogram (color) | {SPLIT} idx={IDX} | y4={y4_val}")

cbar = plt.colorbar(im)
cbar.set_label("Normalized magnitude (0–1)")

plt.tight_layout()
plt.show()
