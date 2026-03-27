import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score

import config as cfg


# Enable importing ../PrototypicalNetwork when running from src/
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PrototypicalNetwork import EpisodeSplitDataset, ProtoDarkNet19  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eval_episodes(
    model: ProtoDarkNet19,
    ds: EpisodeSplitDataset,
    device: torch.device,
    episodes: int,
    n_way: int,
    n_support: int,
    n_query: int,
    sample_with_replacement: bool,
) -> Dict[str, float]:
    model.eval()
    loss_list: List[float] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    with torch.no_grad():
        for _ in range(episodes):
            xs, xq, _ = ds.sample_episode(
                n_way=n_way,
                n_support=n_support,
                n_query=n_query,
                sample_with_replacement=sample_with_replacement,
            )
            xs = xs.to(device, non_blocking=True)
            xq = xq.to(device, non_blocking=True)

            loss, _, y_true, y_pred = model.episode_loss(xs, xq)
            loss_list.append(float(loss.item()))
            all_true.append(y_true.detach().cpu().numpy())
            all_pred.append(y_pred.detach().cpu().numpy())

    y_true_np = np.concatenate(all_true) if all_true else np.array([], dtype=np.int64)
    y_pred_np = np.concatenate(all_pred) if all_pred else np.array([], dtype=np.int64)

    acc = float((y_true_np == y_pred_np).mean()) if y_true_np.size > 0 else 0.0
    f1 = (
        float(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0))
        if y_true_np.size > 0
        else 0.0
    )
    loss_mean = float(np.mean(loss_list)) if loss_list else 0.0
    return {"loss": loss_mean, "acc": acc, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default=cfg.PROTO_DATA_ROOT)
    ap.add_argument("--img_size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--normalize", type=str, default="imagenet", choices=["imagenet", "none"])
    ap.add_argument("--n_way", type=int, default=cfg.PROTO_N_WAY)
    ap.add_argument("--n_support", type=int, default=cfg.PROTO_N_SUPPORT)
    ap.add_argument("--n_query", type=int, default=cfg.PROTO_N_QUERY)
    ap.add_argument("--episodes", type=int, default=cfg.PROTO_EPISODES)
    ap.add_argument("--eval_every", type=int, default=cfg.PROTO_EVAL_EVERY)
    ap.add_argument("--val_episodes", type=int, default=cfg.PROTO_VAL_EPISODES)
    ap.add_argument("--test_episodes", type=int, default=cfg.PROTO_TEST_EPISODES)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--sample_with_replacement", type=int, default=1, choices=[0, 1])
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--save_dir", type=str, default=cfg.CKPT_DIR)
    ap.add_argument("--log_dir", type=str, default=cfg.LOG_DIR)
    ap.add_argument("--early_stop_eval_rounds", type=int, default=0)
    args = ap.parse_args()

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    ds_tr = EpisodeSplitDataset(
        data_root=args.data_root, split="train", image_size=args.img_size, normalize=args.normalize
    )
    ds_va = EpisodeSplitDataset(
        data_root=args.data_root, split="val", image_size=args.img_size, normalize=args.normalize
    )
    ds_te = EpisodeSplitDataset(
        data_root=args.data_root, split="test", image_size=args.img_size, normalize=args.normalize
    )

    print(f"Data root: {args.data_root}")
    print(f"Train class counts: {ds_tr.class_counts}")
    print(f"Val class counts  : {ds_va.class_counts}")
    print(f"Test class counts : {ds_te.class_counts}")
    print("Protocol: train uses train split only, selection uses val split, final report uses test split.")

    model = ProtoDarkNet19().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.log_dir, f"metrics_proto_darknet19_4cls_{stamp}.csv")
    best_ckpt = os.path.join(args.save_dir, f"proto_darknet19_4cls_best_{stamp}.pth")

    fieldnames = [
        "episode",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "val_f1",
        "test_loss",
        "test_acc",
        "test_f1",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

    best_val_f1 = -1.0
    best_episode = 0
    no_imp_rounds = 0
    sample_replace = bool(int(args.sample_with_replacement))

    for ep in range(1, int(args.episodes) + 1):
        model.train()
        xs, xq, _ = ds_tr.sample_episode(
            n_way=int(args.n_way),
            n_support=int(args.n_support),
            n_query=int(args.n_query),
            sample_with_replacement=sample_replace,
        )
        xs = xs.to(device, non_blocking=True)
        xq = xq.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss, tr_acc, _, _ = model.episode_loss(xs, xq)
        loss.backward()
        optimizer.step()
        tr_loss = float(loss.item())

        row = {
            "episode": ep,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": "",
            "val_acc": "",
            "val_f1": "",
            "test_loss": "",
            "test_acc": "",
            "test_f1": "",
        }

        if ep % int(args.eval_every) == 0:
            va = eval_episodes(
                model=model,
                ds=ds_va,
                device=device,
                episodes=int(args.val_episodes),
                n_way=int(args.n_way),
                n_support=int(args.n_support),
                n_query=int(args.n_query),
                sample_with_replacement=sample_replace,
            )
            row["val_loss"] = va["loss"]
            row["val_acc"] = va["acc"]
            row["val_f1"] = va["f1"]

            print(
                f"[VAL] ep{ep:04d}_f1={va['f1']:.4f} ep{ep:04d}_acc={va['acc']:.4f} "
                f"train_loss={tr_loss:.4f}"
            )

            if va["f1"] > best_val_f1:
                best_val_f1 = va["f1"]
                best_episode = ep
                no_imp_rounds = 0
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "best_val_f1": best_val_f1,
                        "best_episode": best_episode,
                        "args": vars(args),
                    },
                    best_ckpt,
                )
            else:
                no_imp_rounds += 1

            if int(args.early_stop_eval_rounds) > 0 and no_imp_rounds >= int(args.early_stop_eval_rounds):
                with open(log_path, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    w.writerow(row)
                print(
                    f"[EarlyStop] no val_f1 improvement for {args.early_stop_eval_rounds} eval rounds."
                )
                break

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)

    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        best_val_f1 = float(ckpt.get("best_val_f1", best_val_f1))
        best_episode = int(ckpt.get("best_episode", best_episode))

    te = eval_episodes(
        model=model,
        ds=ds_te,
        device=device,
        episodes=int(args.test_episodes),
        n_way=int(args.n_way),
        n_support=int(args.n_support),
        n_query=int(args.n_query),
        sample_with_replacement=sample_replace,
    )

    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writerow(
            {
                "episode": "best",
                "train_loss": "",
                "train_acc": "",
                "val_loss": "",
                "val_acc": "",
                "val_f1": best_val_f1,
                "test_loss": te["loss"],
                "test_acc": te["acc"],
                "test_f1": te["f1"],
            }
        )

    print(f"[TEST] best_ep={best_episode:04d} ep{best_episode:04d}_f1={te['f1']:.4f} ep{best_episode:04d}_acc={te['acc']:.4f}")
    print(f"[LOG] {log_path}")
    print(f"[CKPT] {best_ckpt}")


if __name__ == "__main__":
    main()
