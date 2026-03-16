# eval_mlp_4cls.py
import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import src.config as cfg


# ===== Dataset（CWT 紧凑特征，自动扁平化到 [N, D]）=====
class NpyFlatDataset(Dataset):
    def __init__(self, X_path: str, y_path: str, mmap: bool = True):
        self.X = np.load(X_path, mmap_mode='r') if mmap else np.load(X_path)
        if self.X.ndim > 2:
            self.X = self.X.reshape(self.X.shape[0], -1)
        self.X = self.X.astype(np.float32)
        self.y = np.load(y_path).astype(np.int64)

        assert len(self.X) == len(self.y), "X/Y size mismatch"

    def __len__(self): return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.tensor(int(self.y[i]), dtype=torch.long)


# ===== 与训练脚本保持一致的小型 MLP（单隐层）=====
class MLP4(nn.Module):
    def __init__(self, in_dim, hidden=128, p_drop=0.2, num_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        # 期望输入 [B, D]
        return self.net(x)


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
        out = model(x)                    # logits [B, C]
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(y.numpy())
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_labels, all_preds


def main():
    ap = argparse.ArgumentParser()
    # --- 与 eval_darknet_4cls 的参数风格对齐 ---
    ap.add_argument("--data_root",    type=str, default=cfg.DATASET_CWT)
    ap.add_argument("--batch_size",   type=int, default=256)
    ap.add_argument("--checkpoint",   type=str, default="./checkpoints/mlp_cwt_4cls_best.pt")
    ap.add_argument("--num_workers",  type=int, default=4)
    ap.add_argument("--label_prefix", type=str, default="y4")
    ap.add_argument("--num_classes",  type=int, default=4)
    ap.add_argument("--hidden",       type=int, default=128)
    ap.add_argument("--dropout",      type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 数据路径（CWT 特征与 4 类标签）===
    Xte_path = os.path.join(args.data_root, "Xcwt_test.npy")
    yte_path = os.path.join(args.data_root, f"{args.label_prefix}_test.npy")
    ds_te = NpyFlatDataset(Xte_path, yte_path, mmap=True)
    te_loader = DataLoader(
        ds_te, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # === 推断输入维度，并构建模型 ===
    in_dim = int(ds_te.X.shape[1])
    num_classes = int(args.num_classes)
    model = MLP4(in_dim=in_dim, hidden=args.hidden, p_drop=args.dropout, num_classes=num_classes)

    # === 加载权重（兼容 state_dict / DataParallel 前缀）===
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]: v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(device)

    # === 推理评估 ===
    y_true, y_pred = infer_and_eval(model, te_loader, device)
    cm, acc, macro_f1, report = compute_metrics_from_preds(y_true, y_pred, num_classes)

    # === 打印汇总（对齐 eval_darknet_4cls 风格）===
    print("=== Evaluation Summary (4-class, MLP on CWT) ===")
    print(f"Test samples: {len(y_true)}")
    print(f"Accuracy: {acc:.6f}")
    print(f"Macro F1: {macro_f1:.6f}")
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()
