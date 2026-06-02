# MSTAR SAR Target Recognition Dataset

A mirror of the **MSTAR (Moving and Stationary Target Acquisition and Recognition)** public-release dataset for SAR (Synthetic Aperture Radar) automatic target recognition (ATR) research. This repository bundles the two most widely used subsets:

| Subset | Format | Size | Description |
|--------|--------|------|-------------|
| [`soc/`](./soc)         | Optical (`.jpg`)    | ~11 MB  | SOC images rendered to optical format for quick visualization |
| [`soc_complex/`](./soc_complex) | Complex (`.000`/`.025`) | ~923 MB | SOC raw complex SAR data (real + imaginary) |

> The two subsets share **identical filenames and folder structure**, so you can switch between optical and complex inputs by simply pointing your dataloader at the other folder.

---

## Background

MSTAR was collected by the **Air Force Research Laboratory (AFRL)** and the **Defense Advanced Research Projects Agency (DARPA)** in the 1990s. It contains X-band SAR imagery of military ground vehicles measured at varying depression angles. The Standard Operating Conditions (SOC) split — **17° depression for training, 15° for testing** — has become the canonical benchmark for SAR ATR.

Each chip is a 128×128 (or 158×158) SAR image centered on a ground target.

---

## Target Classes (10)

| Class | Vehicle | Notes |
|-------|---------|-------|
| 2S1    | Self-propelled artillery | |
| BMP2   | Infantry fighting vehicle | Multiple serial numbers (SN-9563, SN-C21, SN-812) in test |
| BRDM2  | Amphibious armored scout | |
| BTR60  | Amphibious armored personnel carrier | |
| BTR70  | Amphibious armored personnel carrier | |
| D7     | Bulldozer | |
| T62    | Main battle tank | |
| T72    | Main battle tank | Multiple serial numbers in test |
| ZIL131 | Cargo truck | |
| ZSU234 | Self-propelled anti-aircraft gun | |

---

## Dataset Structure

```
MSTAR/
├── soc/                          # Optical images
│   ├── train/
│   │   ├── 2S1/   (299 files)
│   │   ├── BMP2/  (233 files)
│   │   ├── BRDM2/ (298 files)
│   │   ├── BTR60/ (256 files)
│   │   ├── BTR70/ (233 files)
│   │   ├── D7/    (299 files)
│   │   ├── T62/   (299 files)
│   │   ├── T72/   (232 files)
│   │   ├── ZIL131/(299 files)
│   │   └── ZSU234/(299 files)
│   └── test/
│       ├── 2S1/   (274 files)
│       ├── BMP2/  (195 files, mixed SNs)
│       ├── BRDM2/ (274 files)
│       ├── BTR60/ (195 files)
│       ├── BTR70/ (196 files)
│       ├── D7/    (274 files)
│       ├── T62/   (273 files)
│       ├── T72/   (196 files, mixed SNs)
│       ├── ZIL131/(274 files)
│       └── ZSU234/(274 files)
└── soc_complex/                  # Raw complex SAR data
    ├── train/                    # (same counts as soc/train)
    └── test/                     # (same counts as soc/test)
```

### Sample counts

| Split | soc (optical) | soc_complex |
|-------|---------------|-------------|
| train | 2,747         | 2,747       |
| test  | 2,425         | 2,425       |
| **total** | **5,172**  | **5,172**   |

---

## Filename Convention

Files in `soc_complex/` follow the pattern:

```
<target_serial>_<sequence>.000   (training, depression 17°)
<target_serial>_<sequence>.025   (testing,  depression 15°)
```

Examples:
- `HB14931.000` — training chip
- `HB14931.025` — test chip of the same target at a different depression

The `soc/` optical images share the same base names with `.jpg` extension.

---

## Usage

### Python — quick loader for optical images
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
print(f'{len(dataset)} training samples, {len(dataset.classes)} classes')
```

### Python — loader for complex SAR (.000) data
```python
import numpy as np
from pathlib import Path

def load_complex_chip(path):
    with open(path, 'rb') as f:
        data = np.frombuffer(f.read(), dtype='>f4')
    re, im = data[0::2], data[1::2]
    return (re + 1j * im).reshape(128, 128)  # adjust shape to your dataset
```

---

## Citation

The MSTAR dataset is widely cited in SAR ATR literature. A representative reference:

> Ross, T. D., Worrell, S. W., Velten, V. J., Mossing, J. C., & Bryant, M. L. (1998).
> *Standard SAR ATR evaluation experiments using the MSTAR public release data set*.
> SPIE Conference on Algorithms for Synthetic Aperture Radar Imagery V, 3370, 566–573.

---

## License & Disclaimer

The MSTAR dataset was **publicly released by the U.S. Department of Defense** for research purposes. This repository is a redistribution of that public release to make it easier to access for academic and non-commercial research.

- ✅ Permitted: academic research, educational use, benchmarking
- ❌ Not permitted: military operational use, commercial surveillance, redistribution as a paid product

If you are the rights holder and would like content adjusted or removed, please open an issue.

---

## Repository Info

- **Owner:** [@omae11](https://github.com/omae11)
- **Original source:** U.S. Air Force Research Laboratory (AFRL) MSTAR Public Release
- **Mirror purpose:** easy `git clone` access for researchers
