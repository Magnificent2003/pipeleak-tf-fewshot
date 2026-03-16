import os, sys
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.config as cfg
from src.NpyDataset import NpyDataset
from src.HierarchicalDarknet19 import HierarchicalDarknet19
from sklearn.metrics import f1_score, confusion_matrix, classification_report

# --------------------- Evaluation ---------------------
@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    model.eval()
    all_child_logits = []
    all_y = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, dtype=torch.long, device=device)
        _, child_logits = model(x)
        all_child_logits.append(child_logits)
        all_y.append(y)

    child_logits = torch.cat(all_child_logits, dim=0)
    ys = torch.cat(all_y, dim=0)
    pred = child_logits.argmax(dim=1)

    acc = (pred == ys).float().mean().item()

    y_np = ys.cpu().numpy()
    p_np = pred.cpu().numpy()
    avg = "macro" if num_classes > 2 else "binary"
    f1 = f1_score(y_np, p_np, average=avg, zero_division=0)
    cm = confusion_matrix(y_np, p_np)
    report = classification_report(y_np, p_np, digits=4)

    return acc, f1, cm, report


# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.DATASET_STFT)
    ap.add_argument("--label_prefix", type=str, default="y4")
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_classes", type=int, default=4)
    ap.add_argument("--ckpt", type=str, default="./checkpoints/darknet19_hier_best_20251208-092625.pth")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据
    Xte = os.path.join(args.data_root, f"X_test_stft_{args.img_size}.npy")
    yte = os.path.join(args.data_root, f"{args.label_prefix}_test.npy")
    ds_te = NpyDataset(Xte, yte, normalize="imagenet", memmap=True)
    te_loader = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = HierarchicalDarknet19(num_classes=args.num_classes).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    print(f"Loaded checkpoint: {args.ckpt}")

    # 评估
    acc, f1, cm, report = evaluate(model, te_loader, device, args.num_classes)
    print(f"\n[Test Results] ACC={acc:.4f} | F1={f1:.4f}\n")
    if cm is not None:
        print("Confusion Matrix:\n", cm)
    if report is not None:
        print("\nClassification Report:\n", report)


if __name__ == "__main__":
    main()
    