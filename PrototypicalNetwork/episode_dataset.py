import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class EpisodeSplitDataset:
    """
    Episode sampler over split-specific STFT PNGs and y4 labels.
    - train: png_train + y4_train.npy
    - val:   png_val   + y4_val.npy
    - test:  png_test  + y4_test.npy (fallback: test_val)
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        image_size: int = 224,
        normalize: str = "imagenet",
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unsupported split: {split}")

        if split == "test":
            png_dir = self.data_root / "png_test"
            if not png_dir.exists():
                # Compatible fallback for typo style path naming.
                alt = self.data_root / "test_val"
                png_dir = alt if alt.exists() else png_dir
        else:
            png_dir = self.data_root / f"png_{split}"

        y4_path = self.data_root / f"y4_{split}.npy"
        if not png_dir.exists():
            raise FileNotFoundError(f"Missing image dir for split={split}: {png_dir}")
        if not y4_path.exists():
            raise FileNotFoundError(f"Missing label file for split={split}: {y4_path}")

        y4 = np.load(y4_path).astype(np.int64).reshape(-1)
        class_to_files: Dict[int, List[Path]] = {}

        for i, lab in enumerate(y4.tolist()):
            p = png_dir / f"{split}_{i:06d}.png"
            if p.exists():
                class_to_files.setdefault(int(lab), []).append(p)

        self.class_to_files = {k: sorted(v) for k, v in class_to_files.items() if len(v) > 0}
        self.classes = sorted(self.class_to_files.keys())
        if len(self.classes) == 0:
            raise RuntimeError(f"No valid samples found in split={split} ({png_dir}).")

        self.class_counts = {int(k): int(len(v)) for k, v in self.class_to_files.items()}

        if normalize == "imagenet":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif normalize == "none":
            self.transform = transforms.Compose(
                [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
            )
        else:
            raise ValueError(f"Unsupported normalize mode: {normalize}")

    def _load_image(self, p: Path) -> torch.Tensor:
        with Image.open(p) as img:
            return self.transform(img.convert("RGB"))

    def sample_episode(
        self,
        n_way: int,
        n_support: int,
        n_query: int,
        sample_with_replacement: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        if n_way > len(self.classes):
            raise ValueError(f"n_way={n_way} exceeds split class count={len(self.classes)}")
        if n_way <= 0 or n_support <= 0 or n_query <= 0:
            raise ValueError("n_way/n_support/n_query must be positive.")

        selected = random.sample(self.classes, n_way)
        need = n_support + n_query

        xs_list: List[torch.Tensor] = []
        xq_list: List[torch.Tensor] = []

        for c in selected:
            files = self.class_to_files[c]
            if len(files) >= need:
                picked = random.sample(files, need)
            else:
                if not sample_with_replacement:
                    raise ValueError(
                        f"Class {c} in split={self.split} has {len(files)} samples but need {need}."
                    )
                picked = [random.choice(files) for _ in range(need)]

            support_files = picked[:n_support]
            query_files = picked[n_support:]
            xs_list.append(torch.stack([self._load_image(p) for p in support_files], dim=0))
            xq_list.append(torch.stack([self._load_image(p) for p in query_files], dim=0))

        xs = torch.stack(xs_list, dim=0)
        xq = torch.stack(xq_list, dim=0)
        return xs, xq, selected
