import os
import shutil
import numpy as np

STFT_DIR = "C:\\Users\\36094\\Desktop\\data\\dataset_stft"
OUT_DIR = "C:\\Users\\36094\\Desktop\\data\\mydataset"

class_names = {
    0: "leak",
    1: "valve",
    2: "normal",
    3: "noise"
}

os.makedirs(OUT_DIR, exist_ok=True)

for name in class_names.values():
    os.makedirs(os.path.join(OUT_DIR, name), exist_ok=True)


def process_split(split):

    y = np.load(os.path.join(STFT_DIR, f"y4_{split}.npy"))
    png_dir = os.path.join(STFT_DIR, f"png_{split}")

    for i, label in enumerate(y):

        img_name = f"{split}_{i:06d}.png"
        src = os.path.join(png_dir, img_name)

        if not os.path.exists(src):
            continue

        dst_dir = os.path.join(OUT_DIR, class_names[int(label)])

        dst = os.path.join(dst_dir, img_name)

        shutil.copy(src, dst)


for split in ["train", "val", "test"]:
    process_split(split)

print("完成：图像已经按类别整理")