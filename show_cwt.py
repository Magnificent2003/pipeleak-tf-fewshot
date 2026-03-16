import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== 全局字体：Times New Roman =====
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["mathtext.fontset"] = "stix"
mpl.rcParams["axes.unicode_minus"] = False

import src.config as cfg  # 你的数据构建脚本也是 import config as cfg :contentReference[oaicite:1]{index=1}


# ========= 在这里指定 =========
SPLIT = "train"   # train/val/test
IDX = 3           # 你要看的样本索引
PLOT_MODE = "curves"  # "curves" 或 "heatmap"


def main():
    root = cfg.DATASET_CWT

    X_path = os.path.join(root, f"Xcwt_{SPLIT}.npy")
    y4_path = os.path.join(root, f"y4_{SPLIT}.npy")
    freqs_path = os.path.join(root, "cwt_freqs.npy")

    Xc = np.load(X_path, mmap_mode="r")
    y4 = np.load(y4_path, mmap_mode="r")
    freqs = np.load(freqs_path)  # (S,)

    if IDX < 0 or IDX >= len(Xc):
        raise IndexError(f"IDX={IDX} 超范围：{SPLIT} 共 {len(Xc)} 条")

    y4_val = int(np.asarray(y4[IDX]).reshape(-1)[0])
    print(f"[CWT] split={SPLIT}, idx={IDX}, y4={y4_val}")

    vec = np.asarray(Xc[IDX], dtype=np.float32)  # (4*S,)
    S = len(freqs)
    if vec.size != 4 * S:
        raise ValueError(f"维度不匹配：vec={vec.size}, 但 freqs={S}，期望 vec=4*S={4*S}")

    F = vec.reshape(S, 4)  # (S,4) = [mean, std, q1, q3]（标准化后的）:contentReference[oaicite:2]{index=2}

    stat_names = ["Mean", "Std", "Q1 (25%)", "Q3 (75%)"]

    if PLOT_MODE == "curves":
        fig, ax = plt.subplots(figsize=(6, 6))   # 整体也用正方形
        ax.set_box_aspect(1)                     # 关键：绘图区正方形框
        for j in range(4):
            plt.plot(freqs, F[:, j], linewidth=1.6, label=stat_names[j])
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Standardized feature value")
        plt.xlim(0, 2000)
        plt.ylim(-0.7, 1.2)
        # plt.title(f"CWT features vs Frequency | {SPLIT} idx={IDX} | y4={y4_val}")
        plt.legend()
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.show()

    elif PLOT_MODE == "heatmap":
        plt.figure(figsize=(6.5, 4.5))
        im = plt.imshow(
            F,
            aspect="auto",
            origin="lower",
            extent=[0, 4, freqs.min(), freqs.max()],  # x=4个统计量，y=频率
        )
        plt.xticks([0.5, 1.5, 2.5, 3.5], stat_names, rotation=20, ha="right")
        plt.xlabel("Statistics")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(-0.7, 1.2)
        # plt.title(f"CWT feature map (S×4) | {SPLIT} idx={IDX} | y4={y4_val}")
        cbar = plt.colorbar(im)
        cbar.set_label("Standardized value")
        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("PLOT_MODE must be 'curves' or 'heatmap'")


if __name__ == "__main__":
    main()
