# Align2Gen


## Overview

Automatic polyp segmentation in colonoscopy images is critical for the early diagnosis and treatment of colorectal cancer. However, achieving accurate and robust segmentation is a challenging task due to the significant variations in polyp size, shape, and quantity, particularly in the presence of large, small, or numerous polyps. While recent segmentation methods have achieved notable success, most fail to produce stable results because they do not effectively capture feature interactions across different levels within the image, nor do they integrate both positional and detailed polyp information.

To address these issues, we propose MLDNet (Multi-Level Lesion-aware and Detail-injection Network), a novel approach that introduces a set of specialized modules to enhance feature fusion and segmentation performance.

- Multi-level Position and Detail Fusion (MPDF): Ensures that each feature layer incorporates detailed positional information from all levels, enabling the network to better handle polyp variations.
- Selective Step Feature Aggregation (SSFA): Aggregates features from adjacent layers selectively to refine the feature representation at each level.
- Multi-level Detail Injection (MDI): Performs feature fusion and injects detailed information into the aggregated features, improving the model’s ability to segment complex polyp shapes.

## Usage
### Setup
```
python 3.8
pytorch 1.11.0
cuda 11.3
```

### Downloading necessary data
- downloading training dataset and move it into ```./data/TrainDataset/```. It contains two sub-datasets: Kvasir-SEG (900 train samples)[Link](https://datasets.simula.no/kvasir-seg/) and CVC-ClinicDB (550 train samples)[Link](https://polyp.grand-challenge.org/CVCClinicDB/).
- downloading testing dataset and move it into ```./data/TestDataset/```. It contains five sub-datsets: CVC-300 (60 test samples), CVC-ClinicDB (62 test samples), CVC-ColonDB (380 test samples)[Link](http://vi.cvc.uab.es/colon-qa/cvccolondb/), ETIS-LaribPolypDB (196 test samples)[Link](https://polyp.grand-challenge.org/ETISLarib/), Kvasir (100 test samples).
- downloading PVTv2 weights and and move it into ```./lib/```, which can be found in this download [link](https://github.com/whai362/PVT?tab=readme-ov-file) .

### Train or Test
```
cd MLDNet
python train.py
or
python test.py
```

## Pre-computed maps
They can be found in [Google Drive](https://drive.google.com/drive/folders/1dhHpXMBQRjxARhHEHLRaLpLwIlJT9YE6?usp=drive_link).

##  License
The source code is free for research and education use only. Any commercial use should get formal permission first.

## Acknowledgement
Thanks [Polyp-PVT](https://github.com/DengPingFan/Polyp-PVT) for serving as building blocks of MLDNet.
