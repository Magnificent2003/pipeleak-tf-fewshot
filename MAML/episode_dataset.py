import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def _resolve_image_dir(data_root: Path, split: str) -> Path:
    if split == "train":
        candidates = [data_root / "png_train"]
    elif split == "val":
        candidates = [data_root / "png_val"]
    elif split == "test":
        candidates = [data_root / "png_test", data_root / "test_val", data_root / "png_test_val"]
    else:
        raise ValueError(f"Unsupported split: {split}")

    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find image dir for split={split}. Tried: {[str(p) for p in candidates]}")


def _resolve_split_label_path(data_root: Path, split: str) -> Path:
    p = data_root / f"y4_{split}.npy"
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing label file for split={split}: {p}")


class EpisodicSplitDataset(Dataset):
    """
    Episodic dataset with strict split isolation:
    - train -> png_train + y4_train.npy
    - val   -> png_val   + y4_val.npy
    - test  -> png_test (or test_val fallback) + y4_test.npy
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        n_way: int = 4,
        n_shot: int = 5,
        n_query: int = 15,
        n_episode: int = 100,
        image_size: int = 224,
        normalize: str = "imagenet",
        sample_with_replacement: bool = True,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.n_way = int(n_way)
        self.n_shot = int(n_shot)
        self.n_query = int(n_query)
        self.n_episode = int(n_episode)
        self.sample_with_replacement = bool(sample_with_replacement)

        self.img_dir = _resolve_image_dir(self.data_root, split)
        self.y_path = _resolve_split_label_path(self.data_root, split)
        y = np.load(self.y_path).astype(np.int64).reshape(-1)

        if normalize == "imagenet":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif normalize == "none":
            self.transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
        else:
            raise ValueError(f"Unsupported normalize mode: {normalize}")

        self.class_to_files: Dict[int, List[Path]] = {}
        file_map = self._build_indexed_file_map(y_len=len(y))
        for i, cls in enumerate(y.tolist()):
            p = file_map.get(i, None)
            if p is None:
                continue
            self.class_to_files.setdefault(int(cls), []).append(p)

        self.classes = sorted([k for k, v in self.class_to_files.items() if len(v) > 0])
        if len(self.classes) < self.n_way:
            raise ValueError(f"split={split} has {len(self.classes)} classes, n_way={self.n_way}")

        self.class_counts = {int(k): int(len(v)) for k, v in self.class_to_files.items()}

    def _build_indexed_file_map(self, y_len: int) -> Dict[int, Path]:
        # Preferred naming: split_000000.png; test fallback: test_000000.png
        ext_list: Sequence[str] = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
        split_prefixes = [self.split]
        if self.split == "test":
            split_prefixes = ["test", "test_val", "val"]

        file_map: Dict[int, Path] = {}
        for i in range(y_len):
            found = None
            for pref in split_prefixes:
                stem = f"{pref}_{i:06d}"
                for ext in ext_list:
                    cand = self.img_dir / f"{stem}{ext}"
                    if cand.exists():
                        found = cand
                        break
                if found is not None:
                    break
            if found is not None:
                file_map[i] = found

        if len(file_map) >= y_len:
            return file_map

        all_imgs = sorted([p for p in self.img_dir.iterdir() if p.suffix.lower() in ext_list])
        if len(all_imgs) >= y_len:
            return {i: all_imgs[i] for i in range(y_len)}
        raise FileNotFoundError(
            f"split={self.split}: cannot align image files with labels. "
            f"found={len(all_imgs)} labels={y_len} dir={self.img_dir}"
        )

    def _load_image(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            return self.transform(img.convert("RGB"))

    def __len__(self) -> int:
        return self.n_episode

    def __getitem__(self, idx):
        del idx
        chosen_classes = random.sample(self.classes, self.n_way)
        x_shot, x_query, y_shot, y_query = [], [], [], []
        need = self.n_shot + self.n_query

        for epi_cls, real_cls in enumerate(chosen_classes):
            files = self.class_to_files[real_cls]
            if len(files) >= need:
                picked = random.sample(files, need)
            else:
                if not self.sample_with_replacement:
                    raise ValueError(
                        f"split={self.split} class={real_cls} has {len(files)} < {need} and replacement=False"
                    )
                picked = [random.choice(files) for _ in range(need)]

            shot_files = picked[: self.n_shot]
            query_files = picked[self.n_shot :]

            for p in shot_files:
                x_shot.append(self._load_image(p))
                y_shot.append(epi_cls)
            for p in query_files:
                x_query.append(self._load_image(p))
                y_query.append(epi_cls)

        x_shot_t = torch.stack(x_shot, dim=0)
        x_query_t = torch.stack(x_query, dim=0)
        y_shot_t = torch.tensor(y_shot, dtype=torch.long)
        y_query_t = torch.tensor(y_query, dtype=torch.long)
        return x_shot_t, x_query_t, y_shot_t, y_query_t


def collate_episodes(batch):
    x_shot, x_query, y_shot, y_query = [], [], [], []
    for s, q, ys, yq in batch:
        x_shot.append(s)
        x_query.append(q)
        y_shot.append(ys)
        y_query.append(yq)
    return (
        torch.stack(x_shot, dim=0),
        torch.stack(x_query, dim=0),
        torch.stack(y_shot, dim=0),
        torch.stack(y_query, dim=0),
    )
