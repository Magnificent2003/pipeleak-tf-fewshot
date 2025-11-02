<center><h1>基于时频表征与深度学习的少样本管道漏损声信号识别</h1></center>
少样本、强干扰场景下的**漏/非漏**与**细粒度子类**识别基线与改进方法。仓库覆盖三种输入形式（原始波形、CWT 紧凑特征、STFT 谱图）、多类代表模型以及两项方法：**层级一致性学习（HCL）**与**逐类门控融合（CWGF）**。提供统一的数据构建、训练与评测脚本，支持多次独立划分复现。

---

## 亮点与结论

* **统一协议**：同一训练/验证策略下对比 **6 类代表模型 × 3 类输入**。
* **层级一致性学习（HCL）**：在不损失“漏/非漏”硬指标的同时，提高细粒度子类判别。
* **逐类门控融合（CWGF）**：整合多专家模型优势，平均 **F1 ≈ +1.64%** 相对基线。
* **复现性**：支持 **10 组独立划分**循环实验与统计。

---

## 项目结构

```
main
├─ data/                         # 数据与划分
│  ├─ dataset_cwt/               # CWT 特征数据（.npy文件，由脚本生成）
│  ├─ dataset_simple/            # 原始波形/简单特征（.npy文件，由脚本生成）
│  ├─ dataset_stft/              # STFT 谱图/特征（.npy文件与图像，由脚本生成）
│  ├─ ds_train.xlsx              # 训练集划分（源文件，需要用户自行准备）
│  ├─ lb_train.xlsx              # 训练标签清单（源文件，需要用户自行准备）
│  ├─ nm_parsed.xlsx             # 解析后的元数据（由 parse_nm.py 脚本自动生成）
│  └─ nm_train.xlsx              # 原始清单/命名解析来源（源文件，需要用户自行准备）
│
├─ eval/                         # 评测脚本（读取已训练权重/结果并进行评估）
│  ├─ eval_darknet_2cls.py       # 漏/非漏二分类（Darknet）
│  ├─ eval_darknet_4cls.py       # 细粒度 4 分类（Darknet）
│  ├─ eval_fuse_net.py           # CWGF/多专家融合评测
│  ├─ eval_hier_darknet_4cls.py  # HCL 细粒度 4 分类（Darknet）
│  └─ eval_mlp_4cls.py           # 细粒度 4 分类（MLP）
│
├─ resource/                     # 资源占位
│  ├─ baseline/                  # 基线实验记录（日志/权重）
│  ├─ hcl/                       # HCL实验记录（日志/权重）
│  └─ paral/                     # 平行实验记录（日志/权重）
│
├─ src/                          # 训练与数据构建代码
│  ├─ build_cwt_datasets.py           # 由波形批量生成 CWT 特征
│  ├─ build_datasets.py               # 由波形生成原始一维特征
│  ├─ build_stft_datasets.py          # 由波形生成 STFT 谱图/特征
│  ├─ config.py                       # 全局参数与路径配置
│  ├─ Darknet19.py                    # Darknet-19 模型
│  ├─ HierarchicalDarknet19.py        # 融合 HCL 版本的 Darknet 模型
│  ├─ Gatenet.py                      # 门控融合网络（CWGF）
│  ├─ NpyDataset.py                   # 读写 .npy 数据的 Dataset
│  ├─ parse_nm.py                     # 原始命名/表格解析
│  ├─ run_base_cwt_mlp_2cls.py        # 基线：CWT + MLP（二分类）
│  ├─ run_base_cwt_mlp_4cls.py        # 基线：CWT + MLP（四分类）
│  ├─ run_base_cwt_svm_2cls.py        # 基线：CWT + SVM（二分类）
│  ├─ run_base_cwt_svm_4cls.py        # 基线：CWT + SVM（四分类）
│  ├─ run_base_inception-1d_2cls.py   # 基线：Inception-1D（二分类）
│  ├─ run_base_inception-1d_4cls.py   # 基线：Inception-1D（四分类）
│  ├─ run_base_resnet-1d_2cls.py      # 基线：ResNet-1D（二分类）
│  ├─ run_base_resnet-1d_4cls.py      # 基线：ResNet-1D（四分类）
│  ├─ run_base_stft_resnet_2cls.py    # 基线：STFT-ResNet-1D（二分类）
│  ├─ run_darknet_2cls.py             # 基线：Darknet（二分类）
│  ├─ run_darknet_4cls.py             # 基线：Darknet（四分类）
│  ├─ run_fuse_2cls_network.py        # CWGF 融合（二分类）
│  ├─ run_fuse_4cls_network.py        # CWGF 融合（四分类）
│  ├─ run_fuse_network_beifen.py      # 融合网络备份/调参版本
│  ├─ run_hier_cons_darknet_4cls.py   # HCL + Darknet（四分类）
│  ├─ run_hier_darknet_4cls.py        # HIER + Darknet（四分类）
│  └─ train_and_eval.py               # 统一训练+评测入口
│
├─ .gitignore
└─ README.md
```

---

## 数据准备

1. **原始清单解析**
   将原始文件命名/采样信息整理到 `data/nm_train.xlsx`，使用：

```bash
python src/parse_nm.py
```

2. **划分表**
   使用/编辑 `ds_train.xlsx` 与 `ds_test.xlsx`（可准备 10 份对应 10 折）。
   标签清单在 `lb_train.xlsx` 与 `meta_labels/` 中维护（父-子映射等）。

3. **特征构建**
   根据需要生成 CWT / STFT / 简化特征：

```bash
# 原始特征读取
python src/build_datasets.py

# CWT
python src/build_cwt_datasets.py

# STFT
python src/build_stft_datasets.py
```

> 小贴士：CWT 的母小波、尺度范围，STFT 的窗口/步长、采样率等在脚本 `config.py` 中设置。

---

## 训练与评测

### 基线模型

```bash
# 二分类：Darknet
python src/run_darknet_2cls.py

# 四分类：Darknet
python src/run_darknet_4cls.py

# 二/四分类：1D-ResNet / 1D-Inception / MLP / SVM / STFT-ResNet
python src/run_base_resnet-1d_2cls.py
python src/run_base_inception-1d_4cls.py
python src/run_base_cwt_mlp_4cls.py
python src/run_base_cwt_svm_2cls.py
python src/run_base_stft_resnet_2cls.py
```

### 层级一致性标签（HCL）

```bash
python src/run_hier_darknet_4cls.py --exp runs/hcl_darknet_4cls
```

> 说明：在运行 HCL 模型之前，必须先执行 **1. 原始清单解析**，确保存在具备明确映射关系的层级标签。

### 逐类门控融合（CWGF）

```bash
# 先训练多个“专家模型”（如 Darknet / ResNet / MLP 等），再执行：
python src/run_fuse_4cls_network.py
# 二分类版本：
python src/run_fuse_2cls_network.py
```

> 说明：CWGF 模型训练时，请先确保至少存在两个训练数据同源的模型及训练好的权重。

### 独立评测

```bash
python eval/eval_darknet_2cls.py       --ckpt path/to/ckpt/best.pth
python eval/eval_hier_darknet_4cls.py  --ckpt path/to/ckpt/best.pth
python eval/eval_fuse_net.py           --ckpt path/to/ckpt/best.pth
```

> 说明：各脚本的命令行参数与默认路径以代码内 `config.py` 为准；若脚本通过 `config.py` 控制，请先根据你机器与数据路径完成配置。

---

## 关键脚本功能

| 脚本/模块                                       | 功能要点                                        |
| ------------------------------------------- | ------------------------------------------- |
| `parse_nm.py`                               | 解析原始命名/清单 → 统一的 xlsx / npy 元数据表，供后续构建与划分使用。 |
| `build_datasets.py`                         | 生成原始波形数据的 npy 文件，支持保存为张量加速运算。         |
| `build_cwt_datasets.py`                     | 从波形批量生成 CWT 特征（可选择母小波、尺度、降维/归一化等）。          |
| `build_stft_datasets.py`                    | 生成 STFT 谱图（窗口/步长/频段可配置），支持保存为图像或张量。         |
| `NpyDataset.py`                             | 统一的数据集读取（.npy/张量），配合 DataLoader。            |
| `Darknet19.py` / `HierarchicalDarknet19.py` | 模型定义；后者集成层级一致性损失/约束。                        |
| `Gatenet.py`                                | 逐类门控融合层，实现多专家输出的 class-wise gating。         |
| `run_*`                                     | 按输入类型与模型组合的训练脚本；包括 2/4 类版本。                 |
| `eval_*`                                    | 评测脚本：输出混淆矩阵、精确率/召回率/F1、ROC/PR（具体以实现为准）。     |
| `train_and_eval.py`                         | 单脚本完成训练+评测（如需统一入口）。                         |
| `config.py`                                 | 全局配置（路径、SR、窗口长度、batch size、学习率、类别数、父-子映射等）。 |

---

## 复现实验

Windows（PowerShell）：

```powershell
1..10 | % {
  python src/run_darknet_4cls.py `
    --split data/ds_train_split$_.xlsx `
    --exp runs/darknet4_cls_split$_
}
```

随后使用 `eval/` 脚本批量评测并统计均值/方差。

---

## 配置建议（`config.py`）

* **数据**：`dataset_root`, `sr`（采样率）, `duration`（裁剪时长/帧长），`normalize`，`augment`（可选噪声/混叠）。
* **CWT**：`wavelet`（母小波）、`scales`、`freq_range`、`size`（压缩尺寸）。
* **STFT**：`n_fft`, `win_length`, `hop_length`, `fmin/fmax`、`mel/log`。
* **训练**：`batch_size`, `max_epochs`, `optimizer`, `lr`, `scheduler`, `mixup/cutmix`（如有）。
* **标签**：`num_classes`（2/4）、`hier_map`（父-子映射）、`class_weights`（类不均衡）。

> 以上建议值详见 `config.py`。

## 联系作者

- **作者**：李峙
- **单位**：浙江大学
- **GitHub**：[Magnificent2003](https://github.com/Magnificent2003)
- **邮箱**：1375173994@qq.com
