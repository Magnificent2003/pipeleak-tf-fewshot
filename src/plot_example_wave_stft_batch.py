import os
from pathlib import Path
from typing import Dict, List

import librosa
import matplotlib
import numpy as np
import pandas as pd

import src.config as cfg

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams["axes.unicode_minus"] = False


# idx 使用 Excel 里的 1-based 顺序
IDX_GROUPS: Dict[str, List[int]] = {
    "seepage_leak": [6, 8, 16],
    "valve_leak": [10, 45, 54],
    "normal_non_leak": [14, 19, 23],
    "interference_dominated_non_leak": [15, 24, 27],
}


def normalize_signal(sig: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float32).reshape(-1)
    mu = float(x.mean())
    sd = float(x.std())
    if sd < eps:
        return x - mu
    return (x - mu) / (sd + eps)


def plot_wave(sig_norm: np.ndarray, out_path: Path, fs: float) -> None:
    # 波形图横坐标固定归一化到 0~1
    x_axis = np.linspace(0.0, 1.0, num=sig_norm.shape[0], endpoint=False, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(6.0, 2.8), dpi=300)
    ax.plot(x_axis, sig_norm, color="blue", linewidth=1.0)
    ax.set_xlim(0.0, 1.0)
    # 按需求：不显示横纵坐标
    ax.set_axis_off()

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.tight_layout(pad=0.0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, transparent=True, format="png")
    plt.close(fig)


def plot_stft(sig_norm: np.ndarray, out_path: Path) -> None:
    D = librosa.stft(
        sig_norm.astype(np.float32),
        n_fft=cfg.N_FFT,
        hop_length=cfg.HOP_LENGTH,
        window=cfg.WINDOW,
        center=cfg.CENTER,
    )
    S = np.abs(D)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    S_db = np.clip(S_db, cfg.DB_MIN, cfg.DB_MAX)
    spec01 = (S_db - cfg.DB_MIN) / max(cfg.DB_MAX - cfg.DB_MIN, 1e-12)

    fs = float(getattr(cfg, "FS", 8192))
    freqs = librosa.fft_frequencies(sr=fs, n_fft=cfg.N_FFT)
    times = librosa.frames_to_time(np.arange(spec01.shape[1]), sr=fs, hop_length=cfg.HOP_LENGTH)
    extent = [times.min(), times.max(), freqs.min(), freqs.max()]

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    ax.imshow(
        spec01,
        origin="lower",
        aspect="auto",
        cmap="turbo",
        vmin=0.0,
        vmax=1.0,
        extent=extent,
        interpolation="nearest",
    )
    # 标准正方形显示区域
    ax.set_box_aspect(1)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(fs) / 2.0)
    y_ticks = [0, 1024, 2048, 3072, 4096]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(v) for v in y_ticks])
    ax.set_xticks(np.linspace(0.0, 1.0, num=6))
    ax.tick_params(axis="both", which="major", labelsize=11, length=3, width=0.8, direction="out")
    ax.set_axis_on()
    for sp in ax.spines.values():
        sp.set_visible(True)

    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontname("Times New Roman")

    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0, transparent=True, format="png")
    plt.close(fig)


def main() -> None:
    data_x = Path(cfg.DATA_X)
    if not data_x.exists():
        raise FileNotFoundError(f"DATA_X not found: {data_x}")

    out_root = Path(cfg.EXAMPLE_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    X = pd.read_excel(data_x, header=None, engine="openpyxl").astype(np.float32).to_numpy()
    n = X.shape[0]
    fs = float(getattr(cfg, "FS", 8192))

    print(f"Loaded signals: {X.shape}")
    print(f"Output dir: {out_root}")

    for cls_name, idx_list in IDX_GROUPS.items():
        cls_dir = out_root / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        for idx_1based in idx_list:
            idx0 = int(idx_1based) - 1
            if idx0 < 0 or idx0 >= n:
                print(f"[Skip] {cls_name} idx={idx_1based}: out of range (1~{n})")
                continue

            sig_norm = normalize_signal(X[idx0])
            wave_path = cls_dir / f"idx_{idx_1based:04d}_wave.png"
            stft_path = cls_dir / f"idx_{idx_1based:04d}_stft.png"

            plot_wave(sig_norm, wave_path, fs)
            plot_stft(sig_norm, stft_path)
            print(f"[Saved] {cls_name} idx={idx_1based} -> {wave_path.name}, {stft_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()
