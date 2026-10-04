# MSTAR SAR 目标识别数据集

> MSTAR (Moving and Stationary Target Acquisition and Recognition) 公开数据集的镜像仓库，用于 SAR（合成孔径雷达）自动目标识别 (ATR) 研究。

🌐 **语言**: [English](./README.en.md) · **中文**

本仓库包含两个最常用的子集：

| 子集 | 格式 | 大小 | 说明 |
|------|------|------|------|
| [`soc/`](./soc) | 强度图 (.jpg) | ~11 MB | 由 `soc_complex/` 复数数据检波后生成的实值强度图（158×158，8-bit 灰度），方便直接喂给常规 CV 模型 |
| [`soc_complex/`](./soc_complex) | 复数（扩展名因类别而异，见下） | ~923 MB | SOC 数据集的原始复数 SAR 数据（幅度块 + 相位块），可自行检波得到幅度/强度/相位图 |

> 两个子集 **文件名和目录结构完全一致**，只需把 dataloader 指向另一个根目录，就能在强度图和复数输入之间切换。

---

## 背景

MSTAR 由 **美国空军研究实验室 (AFRL)** 和 **国防高级研究计划局 (DARPA)** 于 1990 年代采集，涵盖 X 波段 SAR 对地面军用车辆的成像。**标准操作条件 (Standard Operating Conditions, SOC) 切分**——**17° 俯仰角训练 / 15° 俯仰角测试**——已成为 SAR ATR 领域的标准基准。

每个切片是 128×128（或 158×158）大小的 SAR 图像，以地面目标为中心。

---

## 目标类别（10 类）

| 类别 | 装备 | 说明 |
|------|------|------|
| 2S1    | 自行火炮 | |
| BMP2   | 步兵战车 | 测试集中含多个序列号（SN-9563、SN-C21、SN-812） |
| BRDM2  | 两栖装甲侦察车 | |
| BTR60  | 两栖装甲运兵车 | |
| BTR70  | 两栖装甲运兵车 | |
| D7     | 推土机 | |
| T62    | 主战坦克 | |
| T72    | 主战坦克 | 测试集中含多个序列号 |
| ZIL131 | 军用卡车 | |
| ZSU234 | 自行防空炮 | |

---

## 数据集结构

```
MSTAR/
├── soc/                          # 检波后的强度图
│   ├── train/
│   │   ├── 2S1/   (299)
│   │   ├── BMP2/  (233)
│   │   ├── BRDM2/ (298)
│   │   ├── BTR60/ (256)
│   │   ├── BTR70/ (233)
│   │   ├── D7/    (299)
│   │   ├── T62/   (299)
│   │   ├── T72/   (232)
│   │   ├── ZIL131/(299)
│   │   └── ZSU234/(299)
│   └── test/
│       ├── 2S1/   (274)
│       ├── BMP2/  (195，含多种序列号)
│       ├── BRDM2/ (274)
│       ├── BTR60/ (195)
│       ├── BTR70/ (196)
│       ├── D7/    (274)
│       ├── T62/   (273)
│       ├── T72/   (196，含多种序列号)
│       ├── ZIL131/(274)
│       └── ZSU234/(274)
└── soc_complex/                  # 原始复数 SAR 数据
    ├── train/                    # (与 soc/train 数量完全一致)
    └── test/                     # (与 soc/test 数量完全一致)
```

### 样本数

| 切分 | soc（强度图） | soc_complex（复数） |
|------|---------------|----------------------|
| train | 2,747 | 2,747 |
| test | 2,425 | 2,425 |
| **合计** | **5,172** | **5,172** |

---

## 文件命名规则

`soc_complex/` 下按 `train/`（17° 俯仰）/ `test/`（15° 俯仰）目录区分训练测试。
**扩展名随目标类别而异**（实测：2S1→`.000`，BMP2→`.000`，BRDM2→`.001`，T72→`.015`，ZSU234→`.026` 等），
请勿用扩展名判断训练/测试切分。

示例：
- `soc_complex/train/2S1/HB14931.000` — 2S1 训练切片
- `soc_complex/test/T72/HB03335.015` — T72 测试切片

`soc/` 中的强度图使用相同的基本文件名，后缀为 `.jpg`（每张 158×158，8-bit 灰度）。

---

## 用法

### Python — 强度图快速加载

```python
from torchvision import datasets, transforms

dataset = datasets.ImageFolder(
    root='soc/train',
    transform=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
)
print(f'{len(dataset)} 个训练样本，{len(dataset.classes)} 个类别')
```

### Python — 复数 SAR 数据加载

```python
import re
import numpy as np

def load_complex_chip(path):
    """读取单个复数切片，返回复数图像（尺寸由文件头决定）。

    文件格式（Phoenix 格式，实测）：
      Phoenix ASCII 文本头（长度由头内 PhoenixHeaderLength 给出，
      实测 1594/1973/1976 字节不等，请勿写死）
      + NumberOfColumns × NumberOfRows 大端 float32 幅度块
      + NumberOfColumns × NumberOfRows 大端 float32 相位块（弧度）
    复数图像 = 幅度 × exp(1j × 相位）。
    注意：不是实部/虚部交错存储，也没有统一的 128×128 尺寸。
    """
    with open(path, 'rb') as f:
        raw = f.read()
    head = raw[:4096].decode('ascii', errors='replace')

    def field(key):
        m = re.search(r'^' + key + r'\s*=\s*([0-9]+)', head, re.M)
        return int(m.group(1))

    hlen = field('PhoenixHeaderLength')
    ncol = field('NumberOfColumns')
    nrow = field('NumberOfRows')
    data = np.frombuffer(raw[hlen:], dtype='>f4')
    n = nrow * ncol
    mag = data[:n].reshape(nrow, ncol)
    pha = data[n:2 * n].reshape(nrow, ncol)
    return mag * np.exp(1j * pha)
```

---

## 引用

MSTAR 数据集在 SAR ATR 文献中被广泛引用。代表性参考文献：

> Ross, T. D., Worrell, S. W., Velten, V. J., Mossing, J. C., & Bryant, M. L. (1998).
> *Standard SAR ATR evaluation experiments using the MSTAR public release data set*.
> SPIE Conference on Algorithms for Synthetic Aperture Radar Imagery V, 3370, 566–573.

---

## 许可与免责声明

MSTAR 数据集由 **美国国防部 (U.S. DoD) 公开发布**，用于研究目的。本仓库是这次公开发布的再分发，旨在方便学术和非商业研究访问。

- ✅ 允许：学术研究、教学用途、基准测试
- ❌ 不允许：军事作战用途、商业监控、作为付费产品再分发

如果您是权利人并希望调整或移除内容，请提 issue。

---

## 仓库信息

- **维护者**: [@omae11](https://github.com/omae11)
- **数据来源**: 美国空军研究实验室 (AFRL) MSTAR Public Release
- **镜像目的**: 让研究者可以轻松 `git clone` 访问
