import os

__CURR__   = os.path.dirname(os.path.abspath(__file__))
__BASE__   = os.path.abspath(os.path.join(__CURR__, ".."))

DATA_DIR   = os.path.join(__BASE__, "data")
DATA_X     = os.path.join(DATA_DIR, "ds_train.xlsx")            # 原始训练数据
DATA_Y     = os.path.join(DATA_DIR, "lb_train.xlsx")            # 原始标签
DATASET    = os.path.join(DATA_DIR, "dataset_simple")           # 制作的数据集目录
DATASET_STFT = os.path.join(DATA_DIR, "dataset_stft")           # STFT 数据集目录
DATASET_CWT  = os.path.join(DATA_DIR, "dataset_cwt")            # CWT 数据集目录
CKPT_DIR   = os.path.join(__BASE__, "checkpoints")              # 默认模型权重保存目录
LOG_DIR    = os.path.join(__BASE__, "logs")                     # 默认日志保存目录
EXAMPLE_DIR = os.path.join(__BASE__, "example")                 # 示例图默认输出目录

# ========== 数据集制作参数 ==========
RATIOS         = (0.7, 0.2, 0.1)                                # train/val/test split ratios
TRAIN_FRACTION = 1.0                                            # Few-shot analysis, default = 1.0
RANDOM_SEED    = 42

SEED_EXP0  = 2016
SEED_EXP1  = 2017
SEED_EXP2  = 2018
SEED_EXP3  = 2019
SEED_EXP4  = 2020                                               # 数据集划分种子，便于复现
SEED_EXP5  = 2021
SEED_EXP6  = 2022
SEED_EXP7  = 2023
SEED_EXP8  = 2024
SEED_EXP9  = 2025
SEED       = SEED_EXP7

# ========== STFT 数据集可配置参数 ==========
SAVE_PNG       = True                         # 是否保存 PNG 方便可视化
IMG_SIZE       = 224                          # 输出给 ResNet 的边长（224或256等）
N_FFT          = 512                          # FFT 点数
HOP_LENGTH     = 128                          # 帧移
WINDOW         = "hann"                       # 窗函数
CENTER         = True                         # librosa.stft(center=True) 默认在两侧补零
DB_MIN, DB_MAX = -80.0, 0.0                   # 对数幅度谱归一化区间（单位：dB）

# ========== CWT 数据集可配置参数 ==========
FS             = 8192                         # 采样率 Hz
WAVELET        = "morl"                       # Morlet 家族
N_SCALES       = 96                           # 尺度数（频率log取样）
FMIN_HZ        = 8.0                          # 最低频
FMAX_HZ        = min(2000.0, FS/2 - 1)        # 最高频（别超过Nyquist）
USE_LOG1P      = True                         # 对|CWT|做 log1p
EPS            = 1e-8



# OUT_DIR  = BASE_DIR / "pipe_dataset_simple" # 输出的本地数据集目录
