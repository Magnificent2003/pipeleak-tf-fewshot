import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import librosa
import config as cfg

os.makedirs(cfg.DATASET_STFT, exist_ok=True)

def load_split(split_name):
    X_path = os.path.join(cfg.DATASET, f"X_{split_name}.npy")
    y_path = os.path.join(cfg.DATASET, f"y_{split_name}.npy")
    X = np.load(X_path)
    y = np.load(y_path)
    return X, y

def stft_to_db(x_1d):
    """
    对单条波形做 STFT -> 幅度 -> dB，并裁剪到 [DB_MIN, DB_MAX]。
    返回二维数组 (freq_bins, time_frames)，dtype=float32。
    """
    # STFT
    D = librosa.stft(x_1d.astype(np.float32), n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
                     window=cfg.WINDOW, center=cfg.CENTER)
    S = np.abs(D)  # 幅度谱
    # 转 dB（以峰值为参考）
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    # 裁剪并线性映射到 [0,1]
    S_db = np.clip(S_db, cfg.DB_MIN, cfg.DB_MAX)
    S_db = (S_db - cfg.DB_MIN) / (cfg.DB_MAX - cfg.DB_MIN)
    return S_db.astype(np.float32)

def spec_to_image(spec_2d, img_size=224):
    """
    把二维谱图 [0,1] -> uint8灰度，缩放到 img_size，再复制为3通道。
    返回形状 (3, img_size, img_size) 的 float32 张量，范围 [0,1]。
    """
    img = Image.fromarray((spec_2d * 255.0).astype(np.uint8))
    # 双线性插值以统一尺寸
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_rgb = Image.merge("RGB", (img, img, img))
    arr = np.asarray(img_rgb, dtype=np.float32) / 255.0  # H×W×3
    arr = np.transpose(arr, (2, 0, 1))                   # 3×H×W
    return arr

def process_split(split_name):
    X, y = load_split(split_name)
    n = X.shape[0]

    X_out = np.zeros((n, 3, cfg.IMG_SIZE, cfg.IMG_SIZE), dtype=np.float32)

    png_dir = os.path.join(cfg.DATASET_STFT, f"png_{split_name}")
    if cfg.SAVE_PNG:
        os.makedirs(png_dir, exist_ok=True)

    for i in tqdm(range(n), desc=f"STFT {split_name}", ncols=80):
        x = X[i]
        x = np.asarray(x).reshape(-1)

        spec_db01 = stft_to_db(x)          # -> [0,1] 2D
        img_chw = spec_to_image(spec_db01, cfg.IMG_SIZE)
        X_out[i] = img_chw

        if cfg.SAVE_PNG:
            png_img = Image.fromarray((spec_db01 * 255.0).astype(np.uint8))
            png_img = png_img.resize((cfg.IMG_SIZE, cfg.IMG_SIZE), Image.BILINEAR)
            png_img.save(os.path.join(png_dir, f"{split_name}_{i:06d}.png"))

    np.save(os.path.join(cfg.DATASET_STFT, f"X_{split_name}_stft_{cfg.IMG_SIZE}.npy"), X_out)
    np.save(os.path.join(cfg.DATASET_STFT, f"y_{split_name}.npy"), y)

    y4_src = os.path.join(cfg.DATASET, f"y4_{split_name}.npy")
    y4_dst = os.path.join(cfg.DATASET_STFT, f"y4_{split_name}.npy")
    np.save(y4_dst, np.load(y4_src))

def main():
    print("输出目录：", cfg.DATASET_STFT)
    for split in ["train", "val", "test"]:
        process_split(split)
    print("全部完成。已生成：")
    print(f" - {os.path.join(cfg.DATASET_STFT, f'X_train_stft_{cfg.IMG_SIZE}.npy')}")
    print(f" - {os.path.join(cfg.DATASET_STFT, f'X_val_stft_{cfg.IMG_SIZE}.npy')}")
    print(f" - {os.path.join(cfg.DATASET_STFT, f'X_test_stft_{cfg.IMG_SIZE}.npy')}")
    print("以及对应标签 y_train/y_val/y_test.npy 和 y4_train/y4_val/y4_test.npy（原样拷贝）")
    if cfg.SAVE_PNG:
        print("另已在 png_train / png_val / png_test 目录下保存可视化 PNG。")

if __name__ == "__main__":
    main()