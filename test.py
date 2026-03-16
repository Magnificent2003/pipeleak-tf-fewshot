import os, numpy as np
import src.config as cfg  # 或 import config as cfg

y4 = np.load(os.path.join(cfg.DATASET, "y4_train.npy"))
print("y4_train unique:", np.unique(y4))
vals, cnts = np.unique(y4, return_counts=True)
print(dict(zip(vals.tolist(), cnts.tolist())))
print("shape:", y4.shape, "dtype:", y4.dtype)
print("first20:", y4[:500].reshape(-1).tolist())
