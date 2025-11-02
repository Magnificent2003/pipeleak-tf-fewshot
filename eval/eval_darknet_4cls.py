import os, sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.config as cfg
from src.Darknet19 import Darknet19
from src.NpyDataset import NpyDataset


def compute_metrics_from_preds(y_true, y_pred, num_classes: int):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, labels=list(range(num_classes)))

    return cm, acc, macro_f1, report


@torch.no_grad()
def infer_and_eval(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        preds = out.argmax(dim=1).cpu().numpy()
        labels = np.asarray(y, dtype=np.int64)
        all_preds.append(preds)
        all_labels.append(labels)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels, all_preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--checkpoint", type=str, default="./checkpoints/darknet19_4cls_best.pth")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--label_prefix", type=str, default="y4")
    ap.add_argument("--num_classes", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 数据路径 ===
    Xte = os.path.join(args.data_root, f"X_test_stft_{args.img_size}.npy")
    yte = os.path.join(args.data_root, f"{args.label_prefix}_test.npy")
    ds_te = NpyDataset(Xte, yte, normalize="imagenet", memmap=True)
    te_loader = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # === 加载权重（兼容 state_dict / DataParallel 前缀 / num_classes 字段） ===
    ckpt = torch.load(args.checkpoint, map_location=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    if "num_classes" in ckpt:
        num_classes = int(ckpt["num_classes"])
    else:
        num_classes = int(np.max(np.load(yte)) + 1)

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

    # === 推理评估 ===
    y_true, y_pred = infer_and_eval(model, te_loader, device)
    cm, acc, macro_f1, report = compute_metrics_from_preds(y_true, y_pred, num_classes)

    # === 打印汇总 ===
    print("=== Evaluation Summary (4-class) ===")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro F1: {macro_f1:.6f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
