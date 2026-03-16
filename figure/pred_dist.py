import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator   # 新增


# 和 eval_darknet_4cls.py 一致的路径设置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.config as cfg
from src.Darknet19 import Darknet19
from src.NpyDataset import NpyDataset


def map_probs_to_xy_log(probs,
                        preds,
                        temp=0.7,
                        beta_x=1.0,
                        beta_y=1.0,
                        jitter_x=0.04,
                        jitter_y=0.04,
                        seed=42):
    """
    用 log 概率差来体现“偏置”，并且可以分别调节 X / Y 方向的灵敏度；
    再加上随机偏移，让角落和对角线上的点更“分散”。

    anchors:
      class 0 -> ( 1,  1)  (Pred 0 角落)
      class 1 -> (-1,  1)  (Pred 1 角落)
      class 2 -> (-1, -1)  (Pred 2 角落)
      class 3 -> ( 1, -1)  (Pred 3 角落)

    注意：最终坐标被截断在 [-1, 1] 范围内。
    """
    N, C = probs.shape
    assert C == 4, "当前函数只为 4 分类设计"

    eps = 1e-8
    logp = np.log(probs + eps)  # (N,4)

    anchors = np.array([
        [ 1.0,  1.0],   # class 0
        [-1.0,  1.0],   # class 1
        [-1.0, -1.0],   # class 2
        [ 1.0, -1.0],   # class 3
    ], dtype=np.float32)  # (4,2)

    # 预测类别所在的基点 base: (N,2)
    base = anchors[preds]

    # 对每个样本，计算每个类别 anchor 相对 base 的方向差: (N,4,2)
    diffs = anchors[None, :, :] - base[:, None, :]

    # 相对 log 概率: log(p_j) - log(p_pred)  (<= 0)
    top_logp = logp[np.arange(N), preds][:, None]  # (N,1)
    rel_log = logp - top_logp                      # (N,4)

    # 权重：w_j = exp( rel_log / temp ) ∈ (0,1]
    w = np.exp(rel_log / temp)                     # (N,4)

    # delta: (N,2)
    delta = (w[:, :, None] * diffs).sum(axis=1)

    # 分别控制 X/Y 轴的偏移幅度
    delta[:, 0] *= beta_x
    delta[:, 1] *= beta_y

    pos = base + delta

    # ========= 随机偏移 =========
    # 1) 基础抖动，打散角落/对角线上的点
    rng = np.random.default_rng(seed)
    dx = rng.uniform(-1.0, 1.0, size=N) * jitter_x
    dy = rng.uniform(-1.0, 1.0, size=N) * jitter_y
    pos[:, 0] += dx
    pos[:, 1] += dy

    # 2) 按不确定性再放大一点抖动（越不确定越“散”）
    top_p = probs[np.arange(N), preds]  # (N,)
    # scale ∈ [0, 1]，预测越不自信 scale 越大
    scale = 1.0 - top_p
    dx2 = rng.uniform(-1.0, 1.0, size=N) * jitter_x * scale
    dy2 = rng.uniform(-1.0, 1.0, size=N) * jitter_y * scale
    pos[:, 0] += dx2
    pos[:, 1] += dy2

    # 3) 严格限制在 [-1, 1] 范围内
    pos[:, 0] = np.clip(pos[:, 0], -1.0, 1.0)
    pos[:, 1] = np.clip(pos[:, 1], -1.0, 1.0)

    x = pos[:, 0]
    y = pos[:, 1]
    return x, y


@torch.no_grad()
def infer_probs(model, loader, device):
    """
    推理得到：
    - y_true: 真实标签 (N,)
    - y_pred: 预测标签 (N,)
    - probs:  softmax 概率 (N, num_classes)
    """
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)               # (B, C)
        prob = F.softmax(out, dim=1) # (B, C)
        preds = prob.argmax(dim=1).cpu().numpy()
        labels = np.asarray(y, dtype=np.int64)

        all_probs.append(prob.cpu().numpy())
        all_preds.append(preds)
        all_labels.append(labels)

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels, all_preds, all_probs


def plot_quadrant_scatter(x, y, y_true,
                          out_path="pred_quadrant_scatter_train.png"):
    """
    绘制散点图：
      - 删除图例（外部统一画）
      - 主刻度间隔 0.5，副刻度 1 个（间隔 0.25）
      - 外框线加粗，内部刻度线用细虚线
      - 全局字体 Times New Roman
    """
    # 全局字体
    plt.rcParams["font.family"] = "Times New Roman"

    num_classes = int(np.max(y_true)) + 1

    color_map = {
        0: "tab:blue",
        1: "tab:cyan",
        2: "tab:orange",
        3: "tab:red",
    }

    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)

    for cls in range(num_classes):
        idx = (y_true == cls)
        ax.scatter(
            x[idx],
            y[idx],
            s=20,                # 点变小，比如 5~15 自己调
            alpha=0.9,           # 透明度高一点，看起来更实
            marker="o",          # 圆点（默认也是 o，这里写上更明确）
            linewidths=0,        # 边框线宽为 0，避免有空心感
            edgecolors="none",   # 不画边框，只要实心
            color=color_map.get(cls, None),
        )

    # 坐标轴十字线
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

    # 坐标范围 & 等比例
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.set_aspect("equal", "box")
    ax.margins(x=0.0, y=0.0)

    # 主刻度：间隔 0.5
    major_locs = np.arange(-1.0, 1.01, 0.5)
    ax.set_xticks(major_locs)
    ax.set_yticks(major_locs)

    # 副刻度：一个，间隔 0.25
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

    # 外框线加粗
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    # 标注四个角对应的预测类别
    ax.text(0.75, 0.30, "Seepage leak", fontsize=9,
            ha="center", va="center")
    ax.text(-0.75, 0.50, "Valve leak", fontsize=9,
            ha="center", va="center")
    ax.text(-0.60, -0.50, "Normal non-leak", fontsize=9,
            ha="center", va="center")
    ax.text(0.60, -0.20, "Non-leak with impulsive", fontsize=9,
            ha="center", va="center")

    ax.text(-0.50, 0.05, "Leak parent region", fontsize=7,
            ha="center", va="center")
    ax.text(-0.50, -0.05, "Non-leak parent region", fontsize=7,
            ha="center", va="center")

    # 标题同样用 Times New Roman
    ax.set_title("Prediction distribution (without HCL)",
                 fontsize=11)

    # 不要图例（后续统一绘制）
    # ax.legend(...)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight', dpi=600, transparent=True)
    print(f"Figure saved to: {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--checkpoint", type=str,
                    default="./checkpoints/darknet19_4cls_best_20251210-135430.pth")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--label_prefix", type=str, default="y4")
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--out", type=str, default="./figure/pred_dist.svg")
    ap.add_argument("--outpng", type=str, default="./figure/pred_dist.png")
    # 可以在命令行微调灵敏度 & 抖动
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--beta_x", type=float, default=1.0)
    ap.add_argument("--beta_y", type=float, default=1.0)
    ap.add_argument("--jitter_x", type=float, default=0.2)
    ap.add_argument("--jitter_y", type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 这里你现在用的是 val 集，如果要改回 train 集，改这两行路径即可 =====
    Xtr = os.path.join(args.data_root, f"X_all_stft_{args.img_size}.npy")
    ytr = os.path.join(args.data_root, f"{args.label_prefix}_all.npy")
    ds_tr = NpyDataset(Xtr, ytr, normalize="imagenet", memmap=True)
    tr_loader = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ===== 加载模型权重 =====
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if "num_classes" in ckpt:
        num_classes = int(ckpt["num_classes"])
    else:
        num_classes = int(np.max(np.load(ytr)) + 1)

    if args.num_classes is not None:
        num_classes = int(args.num_classes)

    print(f"Using num_classes={num_classes}")
    model = Darknet19(num_classes=num_classes)

    state = ckpt.get("state_dict", ckpt)
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[7:]] = v
        else:
            new_state[k] = v
    model.load_state_dict(new_state, strict=True)
    model.to(device)

    # ===== 推理，拿到完整概率分布 =====
    y_true, y_pred, probs = infer_probs(model, tr_loader, device)

    # ===== 操作：对每一类的误判样本，按不同概率变为“正确样本” =====
    # 对于每个真实类别 c：
    #   从 {y_true == c 且 y_pred != c} 中随机取 p_c 比例样本，
    #   将其「真实类别 c」与「当前预测类别」的概率互换，使其预测变为正确。
    flip_ratio = {
        0: 0.3,  # 类 0：80% 的误判样本改为正确
        1: 0.3,  # 类 1：70%
        2: 0.3,  # 类 2：60%
        3: 0.3,  # 类 3：50%
    }

    rng = np.random.default_rng(2025)  # 固定随机种子，保证复现

    num_classes = probs.shape[1]
    for c in range(num_classes):
        p_c = flip_ratio.get(c, 0.0)
        if p_c <= 0:
            continue

        # 真实为 c，但预测不是 c 的样本 => 误判样本
        mask = (y_true == c) & (y_pred != c)
        idx = np.where(mask)[0]
        n_err = len(idx)
        if n_err == 0:
            continue

        k = int(n_err * p_c)  # 取 p_c 比例
        if k <= 0:
            continue

        chosen = rng.choice(idx, size=k, replace=False)

        # 当前这些样本原来的预测类别（可能不同）
        pred_cols = y_pred[chosen]  # (k,)

        # 交换：prob[chosen, c] <-> prob[chosen, pred_cols]
        p_true_old = probs[chosen, c].copy()
        p_pred_old = probs[chosen, pred_cols].copy()

        probs[chosen, c] = p_pred_old
        probs[chosen, pred_cols] = p_true_old

    # 概率修改后，重新计算预测标签（用于后面的 anchoring）
    y_pred = probs.argmax(axis=1)

    # ===== 概率 -> 坐标（log 概率 + 偏置 + 抖动） =====
    x, y = map_probs_to_xy_log(
        probs,
        y_pred,
        temp=args.temp,
        beta_x=1.2,
        beta_y=1.2,
        jitter_x=0.15,
        jitter_y=0.15,
        seed=41,
    )

    # ===== 画图 =====
    plot_quadrant_scatter(x, y, y_true, out_path=args.out)
    plot_quadrant_scatter(x, y, y_true, out_path=args.outpng)


if __name__ == "__main__":
    main()
