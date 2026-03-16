import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
from textwrap import wrap

# 全局字体
mpl.rcParams['font.family'] = 'Times New Roman'

# ================== 数据 ==================
data_AB = np.array([
    [0.46, 0.54],
    [0.50, 0.50],
    [0.55, 0.45],
    [0.57, 0.43]
])

data_CD = np.array([
    [0.80, 0.20],
    [0.35, 0.65],
    [0.50, 0.50],
    [0.25, 0.75]
])

data_EF = np.array([
    [0.90, 0.10],
    [0.70, 0.30],
    [0.82, 0.18],
    [0.88, 0.12]
])

row_labels = ["Impulsive NL", "Normal NL", "Valve leak", "Seepage leak"]
col_labels_list = [
    ["(STFT)\nDarkNet (4-class)", "(STFT)\nResNet (4-class)"],
    ["(STFT)\nDarkNet + HCL", "(STFT)\nResNet (4-class)"],
    ["(STFT)\nDarkNet + HCL", "(CWT)\nMLP (4-class)"]
]
titles = ["Uniform weights (degenerate fusion)", "Structured weights (effective gating)", "Biased weights (expert dominance)"]
matrices = [data_AB, data_CD, data_EF]


# 每个 panel 下方两行小字（先用 XXX 占位）
panel_texts = [
    [r"ΔF1 (experts) < 5%",    r"Gain over best expert ≈ -2.04% (degenerate)"],
    [r"ΔF1 (experts) < 3%",    r"Gain over best expert ≈ 6.39%"],
    [r"ΔF1 (experts) ≈ 7%",    r"Gain over best expert ≈ -1.23% (dominated)"],
]

# 可选：x 轴标签自动换行
def wrap_labels(labels, width=8):
    return ['\n'.join(wrap(str(s), width)) for s in labels]

# 统一 0~1 色标
norm = Normalize(vmin=0.0, vmax=1.0)

# ================== 画图 ==================
fig, axes = plt.subplots(1, 3, figsize=(10, 7.2))

for i, (ax, data, cols, title, txt) in enumerate(
        zip(axes, matrices, col_labels_list, titles, panel_texts)):

    im = ax.imshow(
        data,
        cmap="Blues",
        norm=norm,
        aspect="auto",
        interpolation="bilinear"
    )

    # ---- x 轴在底部 ----
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels(cols, fontsize=10)
    ax.tick_params(axis='x',
                   top=False, bottom=True,
                   labeltop=False, labelbottom=True)
    ax.xaxis.set_ticks_position('bottom')
    # ax.spines['top'].set_visible(False)

    # ---- y 轴：只在第一个 panel 显示文字 ----
    ax.set_yticks(np.arange(len(row_labels)))
    if i == 0:
        ax.set_yticklabels(row_labels, fontsize=10)
    else:
        ax.set_yticklabels([""] * len(row_labels))

    ax.set_title(
        title,
        fontsize=11.5,
        fontweight='bold',  # 加粗
        pad=12              # 标题与图之间的距离（默认一般是 6 左右）
    )

    # ---- 每个 panel 下方两行小字，行距缩小 ----
    line1, line2 = txt

    # 第一行：ΔF1 (experts)
    ax.text(0.5, -0.14, line1,          # 原来大概是 -0.18
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=12, color='dimgray')

    # 第二行：Gain over best expert
    ax.text(0.5, -0.20, line2,          # 原来大概是 -0.33
            transform=ax.transAxes,
            ha='center', va='top',
            fontsize=10, color='dimgray')

# 调整子图间距：给上标题和下灰字都留空间
plt.subplots_adjust(
    left=0.11,   # 视情况可微调
    right=0.88,
    bottom=0.24,
    top=0.88,
    wspace=0.35
)

# 右侧颜色条
cbar_ax = fig.add_axes([0.92, 0.24, 0.015, 0.64])
cbar = fig.colorbar(im, cax=cbar_ax)
cbar.set_label("Weight", rotation=270, labelpad=12, fontsize=11)

# 保存为 SVG：不要再用 bbox_inches='tight'，否则容易把上下裁掉
fig.savefig("./figure/cwgf_experts_heatmaps.svg", format="svg")
fig.savefig("./figure/cwgf_experts_heatmaps.png", bbox_inches='tight', dpi=600, transparent=True)

plt.show()
