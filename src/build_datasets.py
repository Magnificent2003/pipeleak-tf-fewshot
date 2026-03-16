import os
import numpy as np
import pandas as pd
import config as cfg

def stratified_split(y, ratios, seed):
    """按 0/1 分层切分，返回 train/val/test 的索引"""
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    train, val, test = [], [], []
    for cls in [0, 1]:
        idx = idx_all[y == cls]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(np.floor(ratios[0] * n))
        n_va = int(np.floor(ratios[1] * n))
        train.append(idx[:n_tr])
        val.append(idx[n_tr:n_tr+n_va])
        test.append(idx[n_tr+n_va:])
    return np.concatenate(train), np.concatenate(val), np.concatenate(test)

def main():
    os.makedirs(cfg.DATASET, exist_ok=True)

    # ===== 1) 读取特征与二分类标签（Excel）并分层切分 =====
    X = pd.read_excel(cfg.DATA_X, header=None, engine="openpyxl").astype(np.float32).to_numpy()
    y = pd.read_excel(cfg.DATA_Y, header=None, engine="openpyxl").to_numpy().reshape(-1).astype(np.int64)

    tr_idx, va_idx, te_idx = stratified_split(y, cfg.RATIOS, seed=cfg.SEED)

    # 打印分布便于核查
    def dist(arr): 
        u,c = np.unique(arr, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))
    print("数据集按", cfg.RATIOS, "比例切分完成，种子为", cfg.SEED)

    np.save(os.path.join(cfg.DATASET, "X_train.npy"), X[tr_idx])
    np.save(os.path.join(cfg.DATASET, "y_train.npy"), y[tr_idx])
    np.save(os.path.join(cfg.DATASET, "X_val.npy"),   X[va_idx])
    np.save(os.path.join(cfg.DATASET, "y_val.npy"),   y[va_idx])
    np.save(os.path.join(cfg.DATASET, "X_test.npy"),  X[te_idx])
    np.save(os.path.join(cfg.DATASET, "y_test.npy"),  y[te_idx])

    print("已完成二分类数据集划分，保存至：", cfg.DATASET)

    # 同步保存索引
    np.save(os.path.join(cfg.DATASET, "train_idx.npy"), tr_idx)
    np.save(os.path.join(cfg.DATASET, "val_idx.npy"),   va_idx)
    np.save(os.path.join(cfg.DATASET, "test_idx.npy"),  te_idx)

    # ===== 2) 基于 datasets1 的索引 + nm_parsed.xlsx 生成四分类标签 =====
    nm_xlsx_path = os.path.join(cfg.DATA_DIR, "nm_parsed.xlsx")
    df = pd.read_excel(nm_xlsx_path, engine="openpyxl")

    top = pd.to_numeric(df["top_id"], errors="coerce").fillna(-1).astype("int64").to_numpy()
    sub = pd.to_numeric(df["nonleak_sub"], errors="coerce").fillna(-1).astype("int64").to_numpy()

    N = len(top)

    # 0: 渗漏(top==1), 1: 阀门漏水(top==2), 2: 非漏-常规(sub==0), 3: 非漏-干扰(sub==1)
    y4_all = np.full(N, -1, dtype=np.int64)
    y4_all[top == 1] = 0
    y4_all[top == 2] = 1
    y4_all[(top == 0) & (sub == 0)] = 2
    y4_all[(top == 0) & (sub == 1)] = 3

    for name, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        bad = idx[y4_all[idx] < 0]
        if bad.size:
            raise ValueError(f"{name} 集合存在无法归类的样本: {bad[:10]}")

    np.save(os.path.join(cfg.DATASET, "y4_train.npy"), y4_all[tr_idx])
    np.save(os.path.join(cfg.DATASET, "y4_val.npy"),   y4_all[va_idx])
    np.save(os.path.join(cfg.DATASET, "y4_test.npy"),  y4_all[te_idx])

    print("已完成四分类数据集划分，保存至：", cfg.DATASET)

    print("train:", X[tr_idx].shape, dist(y[tr_idx]), dist(y4_all[tr_idx]))
    print("val  :", X[va_idx].shape, dist(y[va_idx]), dist(y4_all[va_idx]))
    print("test :", X[te_idx].shape, dist(y[te_idx]), dist(y4_all[te_idx]))

if __name__ == "__main__":
    main()
