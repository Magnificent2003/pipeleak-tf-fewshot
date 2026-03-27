import argparse
import copy
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from protonets.models.factory import get_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_default_dataset_stft() -> Path:
    here = Path(__file__).resolve().parent
    cands = [
        here.parent / "data" / "dataset_stft",
        here / "data" / "dataset_stft",
    ]
    for p in cands:
        if p.exists():
            return p
    return cands[0]


class SplitFewShotDataset:
    def __init__(self, dataset_stft: Path, split: str, image_size: int = 224):
        self.dataset_stft = Path(dataset_stft)
        self.split = split
        self.png_dir = self.dataset_stft / f"png_{split}"
        self.y4_path = self.dataset_stft / f"y4_{split}.npy"

        if not self.png_dir.exists():
            raise FileNotFoundError(f"Missing split png dir: {self.png_dir}")
        if not self.y4_path.exists():
            raise FileNotFoundError(f"Missing split label file: {self.y4_path}")

        y4 = np.load(self.y4_path).astype(np.int64).reshape(-1)
        class_to_files: Dict[int, List[Path]] = {}
        for i, y in enumerate(y4.tolist()):
            p = self.png_dir / f"{split}_{i:06d}.png"
            if p.exists():
                class_to_files.setdefault(int(y), []).append(p)

        self.class_to_files = {k: sorted(v) for k, v in class_to_files.items() if len(v) > 0}
        self.classes = sorted(self.class_to_files.keys())
        if len(self.classes) == 0:
            raise RuntimeError(f"No valid images found for split={split} under {self.png_dir}")

        self.class_counts = {int(k): int(len(v)) for k, v in self.class_to_files.items()}

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def _load_image(self, path: Path) -> torch.Tensor:
        with Image.open(path) as img:
            return self.transform(img.convert("RGB"))

    def sample_episode(
        self,
        n_way: int,
        n_support: int,
        n_query: int,
    ) -> Dict[str, torch.Tensor]:
        if n_way > len(self.classes):
            raise ValueError(
                f"[{self.split}] n_way={n_way} exceeds class count={len(self.classes)}"
            )
        chosen_classes = random.sample(self.classes, n_way)
        need = n_support + n_query

        xs: List[torch.Tensor] = []
        xq: List[torch.Tensor] = []
        for cls in chosen_classes:
            files = self.class_to_files[cls]
            if len(files) >= need:
                picked = random.sample(files, need)
            else:
                # Small class fallback: sample with replacement (still split-pure, no leakage)
                picked = [random.choice(files) for _ in range(need)]

            support_paths = picked[:n_support]
            query_paths = picked[n_support:]

            support_tensors = [self._load_image(p) for p in support_paths]
            query_tensors = [self._load_image(p) for p in query_paths]
            xs.append(torch.stack(support_tensors, dim=0))
            xq.append(torch.stack(query_tensors, dim=0))

        return {
            "xs": torch.stack(xs, dim=0),  # [n_way, n_support, C, H, W]
            "xq": torch.stack(xq, dim=0),  # [n_way, n_query, C, H, W]
            "classes": chosen_classes,
        }


def one_episode(
    model: torch.nn.Module,
    dataset: SplitFewShotDataset,
    n_way: int,
    n_support: int,
    n_query: int,
    device: torch.device,
    optimizer: optim.Optimizer = None,
) -> Tuple[float, float]:
    sample = dataset.sample_episode(n_way=n_way, n_support=n_support, n_query=n_query)
    sample["xs"] = sample["xs"].to(device, non_blocking=True)
    sample["xq"] = sample["xq"].to(device, non_blocking=True)

    train_mode = optimizer is not None
    if train_mode:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, out = model.loss(sample)
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        with torch.no_grad():
            loss, out = model.loss(sample)

    return float(out["loss"]), float(out["acc"])


def evaluate_episodes(
    model: torch.nn.Module,
    dataset: SplitFewShotDataset,
    n_way: int,
    n_support: int,
    n_query: int,
    episodes: int,
    device: torch.device,
) -> Dict[str, float]:
    losses: List[float] = []
    accs: List[float] = []
    for _ in range(episodes):
        loss, acc = one_episode(
            model=model,
            dataset=dataset,
            n_way=n_way,
            n_support=n_support,
            n_query=n_query,
            device=device,
            optimizer=None,
        )
        losses.append(loss)
        accs.append(acc)

    arr_loss = np.array(losses, dtype=np.float64)
    arr_acc = np.array(accs, dtype=np.float64)
    std = float(arr_acc.std(ddof=1)) if arr_acc.size > 1 else 0.0
    se = float(std / math.sqrt(max(1, arr_acc.size)))
    ci95 = float(1.96 * se)
    return {
        "loss_mean": float(arr_loss.mean()),
        "acc_mean": float(arr_acc.mean()),
        "acc_std": std,
        "acc_se": se,
        "acc_ci95": ci95,
        "episodes": int(episodes),
    }


def save_csv(path: Path, rows: List[Dict[str, float]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Strict no-leak ProtoNet train/val/test split runner."
    )
    ap.add_argument(
        "--dataset_stft",
        type=str,
        default=str(resolve_default_dataset_stft()),
        help="Path to dataset_stft containing png_{train,val,test} and y4_{train,val,test}.npy",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="protonet_darknet19",
        choices=["protonet_darknet19", "protonet_resnet18", "protonet_conv"],
    )
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--train_episodes", type=int, default=5000)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--val_episodes", type=int, default=300)
    ap.add_argument("--test_episodes", type=int, default=1000)
    ap.add_argument("--n_way", type=int, default=4)
    ap.add_argument("--n_support", type=int, default=5)
    ap.add_argument("--n_query", type=int, default=5)
    ap.add_argument(
        "--test_n_support",
        type=int,
        default=None,
        help="If set, override n_support during final test episodes.",
    )
    ap.add_argument(
        "--test_n_query",
        type=int,
        default=None,
        help="If set, override n_query during final test episodes.",
    )
    ap.add_argument("--save_ckpt", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output dir. Default: PROTOTYPICAL/runs/no_leak_<timestamp>",
    )
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    default_out = Path(__file__).resolve().parent / "runs" / f"no_leak_{args.model}_{ts}"
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_stft = Path(args.dataset_stft).resolve()
    ds_train = SplitFewShotDataset(dataset_stft, "train", image_size=args.image_size)
    ds_val = SplitFewShotDataset(dataset_stft, "val", image_size=args.image_size)
    ds_test = SplitFewShotDataset(dataset_stft, "test", image_size=args.image_size)

    print("==== Strict No-Leak ProtoNet ====")
    print(f"dataset_stft : {dataset_stft}")
    print(f"model        : {args.model}")
    print(f"device       : {device}")
    print(f"seed         : {args.seed}")
    print(f"train split  : {ds_train.class_counts}")
    print(f"val split    : {ds_val.class_counts}")
    print(f"test split   : {ds_test.class_counts}")
    print("NOTE: Train uses only train split; model selection uses val split; final report uses test split.")

    if args.model == "protonet_conv":
        model = get_model(args.model, x_dim=(3, args.image_size, args.image_size), hid_dim=64, z_dim=64)
    else:
        model = get_model(args.model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history_rows: List[Dict[str, float]] = []
    best_val_acc = -1.0
    best_state = None
    best_episode = 0
    stale_rounds = 0
    tic = time.time()

    for ep in range(1, args.train_episodes + 1):
        tr_loss, tr_acc = one_episode(
            model=model,
            dataset=ds_train,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            device=device,
            optimizer=optimizer,
        )

        row = {
            "episode": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_acc": "",
            "val_loss": "",
            "best_val_acc": best_val_acc if best_val_acc >= 0 else "",
            "stale_rounds": stale_rounds,
        }

        if ep % args.eval_every == 0:
            val_stats = evaluate_episodes(
                model=model,
                dataset=ds_val,
                n_way=args.n_way,
                n_support=args.n_support,
                n_query=args.n_query,
                episodes=args.val_episodes,
                device=device,
            )
            val_acc = float(val_stats["acc_mean"])
            val_loss = float(val_stats["loss_mean"])
            row["val_acc"] = val_acc
            row["val_loss"] = val_loss

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = copy.deepcopy(model.state_dict())
                best_episode = ep
                stale_rounds = 0
            else:
                stale_rounds += 1

            row["best_val_acc"] = best_val_acc
            row["stale_rounds"] = stale_rounds

            elapsed = time.time() - tic
            print(
                f"[ep {ep:5d}] train_acc={tr_acc:.4f} val_acc={val_acc:.4f} "
                f"best_val={best_val_acc:.4f} stale={stale_rounds}/{args.patience} "
                f"elapsed={elapsed/60.0:.1f}m"
            )
            if stale_rounds >= args.patience:
                print("Early stopping triggered.")
                history_rows.append(row)
                break

        history_rows.append(row)

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_episode = len(history_rows)
        val_stats = evaluate_episodes(
            model=model,
            dataset=ds_val,
            n_way=args.n_way,
            n_support=args.n_support,
            n_query=args.n_query,
            episodes=args.val_episodes,
            device=device,
        )
        best_val_acc = float(val_stats["acc_mean"])

    model.load_state_dict(best_state)

    test_n_support = args.test_n_support if args.test_n_support is not None else args.n_support
    test_n_query = args.test_n_query if args.test_n_query is not None else args.n_query
    test_stats = evaluate_episodes(
        model=model,
        dataset=ds_test,
        n_way=args.n_way,
        n_support=test_n_support,
        n_query=test_n_query,
        episodes=args.test_episodes,
        device=device,
    )

    csv_path = out_dir / "train_trace.csv"
    save_csv(
        csv_path,
        history_rows,
        fieldnames=[
            "episode",
            "train_loss",
            "train_acc",
            "val_acc",
            "val_loss",
            "best_val_acc",
            "stale_rounds",
        ],
    )

    ckpt_path = out_dir / f"{args.model}_best.pth"
    if int(args.save_ckpt) == 1:
        torch.save(best_state, ckpt_path)

    summary = {
        "timestamp": ts,
        "dataset_stft": str(dataset_stft),
        "model": args.model,
        "device": str(device),
        "seed": int(args.seed),
        "n_way": int(args.n_way),
        "n_support_train_val": int(args.n_support),
        "n_query_train_val": int(args.n_query),
        "n_support_test": int(test_n_support),
        "n_query_test": int(test_n_query),
        "train_episodes_requested": int(args.train_episodes),
        "episodes_ran": int(len(history_rows)),
        "eval_every": int(args.eval_every),
        "val_episodes": int(args.val_episodes),
        "test_episodes": int(args.test_episodes),
        "best_episode": int(best_episode),
        "best_val_acc": float(best_val_acc),
        "test_loss_mean": float(test_stats["loss_mean"]),
        "test_acc_mean": float(test_stats["acc_mean"]),
        "test_acc_std": float(test_stats["acc_std"]),
        "test_acc_se": float(test_stats["acc_se"]),
        "test_acc_ci95": float(test_stats["acc_ci95"]),
        "split_class_counts": {
            "train": ds_train.class_counts,
            "val": ds_val.class_counts,
            "test": ds_test.class_counts,
        },
        "no_leakage_protocol": {
            "train_source": "png_train + y4_train.npy only",
            "model_selection_source": "png_val + y4_val.npy only",
            "final_test_source": "png_test + y4_test.npy only",
        },
        "files": {
            "train_trace_csv": str(csv_path),
            "best_ckpt": str(ckpt_path) if int(args.save_ckpt) == 1 else "",
        },
    }

    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n==== Final Test (Unseen Split) ====")
    print(f"best_episode : {best_episode}")
    print(f"best_val_acc : {best_val_acc:.4f}")
    print(
        "test_acc     : "
        f"{test_stats['acc_mean']:.4f} ± {test_stats['acc_std']:.4f} "
        f"(95% CI ± {test_stats['acc_ci95']:.4f}, episodes={args.test_episodes})"
    )
    print(f"saved        : {summary_path}")


if __name__ == "__main__":
    main()
