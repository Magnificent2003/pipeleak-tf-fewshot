import os
import numpy as np
import matplotlib.pyplot as plt

import src.config as cfg



# ===== 读取一条样本（默认第 0 条，可改 idx）=====
idx = 3

# 你要求“读取他们的 X_train.npy”，这里默认从 cfg.DATASET 目录取
dataset_dir = getattr(cfg, "DATASET", None)
if dataset_dir is None:
    raise AttributeError("config 中找不到 DATASET，请确认 X_train.npy 所在目录并在 config 里配置 DATASET。")

X_path = os.path.join(dataset_dir, "X_train.npy")
y4_path = os.path.join(dataset_dir, "y4_train.npy")

X = np.load(X_path)
y4 = np.load(y4_path)

sig = np.asarray(X[idx]).reshape(-1).astype(np.float32)
y4_val = int(y4[idx])

# ===== 绘图开始前打印 y4 =====
print(f"[Waveform] idx={idx}, y4={y4_val}")

# ===== 时间轴 =====
fs = getattr(cfg, "FS", None)
sig = sig - sig.mean()

if fs is not None and fs > 0:
    t = np.arange(len(sig)) / float(fs)
    xlab = "Time (s)"
else:
    t = np.arange(len(sig))
    xlab = "Sample Index"

# ===== 绘图 =====
fig = plt.figure(figsize=(6, 4))

ax = fig.add_axes([0, 0, 1, 1])  # 充满画布，避免留白
ax.plot(t, sig, linewidth=1.2)

# 关掉所有坐标轴元素（刻度、刻度线、标签）
ax.set_axis_off()

# 只保留一个外框（并可加粗）
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_linewidth(2.0)



plt.show()
