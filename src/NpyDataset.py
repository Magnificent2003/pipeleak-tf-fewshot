import torch
import numpy as np
from torch.utils.data import Dataset

class NpyDataset(Dataset):
    """X: (N, 3, H, W) float32 in [0,1]; y: (N,) int"""
    def __init__(self, x_path: str, y_path: str, normalize: str = "imagenet", memmap=True):
        self.X = np.load(x_path, mmap_mode="r" if memmap else None)
        self.y = np.load(y_path)
        assert self.X.ndim == 4 and self.X.shape[1] == 3, f"X shape should be (N,3,H,W), got {self.X.shape}"
        assert len(self.X) == len(self.y), "X and y length mismatch"

        if normalize == "imagenet":
            self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
            self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
        elif normalize == "half":
            self.mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
            self.std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
        else:
            self.mean = None; self.std = None

    def __len__(self): return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]                          # (3,H,W)
        if self.mean is not None:
            x = (x - self.mean) / self.std
        x = torch.from_numpy(np.ascontiguousarray(x))  # float32
        y = int(self.y[idx])
        return x, y
