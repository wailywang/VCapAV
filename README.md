# VCapAV

**VCapAV** is an audio-visual deepfake detection benchmark and toolkit, featuring three key models: **AASIST**, **ResNet**, and **LCNN**. The benchmark supports both unimodal and cross-modal forgery detection using spatial frequency features, log-Fbank spectrograms, and raw waveform encodings.

## Dataset Overview

| Category                | Number of Clips | Total Duration (hours) |
|-------------------------|----------------:|------------------------:|
| Real Video              | 14,923          | 41.45                   |
| Fake Video              | 242             | 0.67                    |
| Real Audio              | 14,923          | 41.45                   |
| Fake Audio              | 74,615          | 207.26                  |
| Real Video + Real Audio | 14,923          | 41.45                   |
| Real Video + Fake Audio | 74,615          | 207.26                  |
| Fake Video + Real Audio | 242             | 0.67                    |
| Fake Video + Fake Audio | 1,210           | 3.36                    |
| **Total**               | **90,990**      | **252.75**              |

For most audio-visual detection applications, we recommend starting with the `ResNet + LCNN` ensemble.

> **Note:** All reported results are based on the closed-world AV-deepfake task using the Wan2.1/Kling/CogVideo benchmark on the `dev` split.

---

## Paper

```bibtex
@inproceedings{wang2025vcapav,
  title={VCapAV: A Video-Caption Based Audio-Visual Deepfake Detection Dataset},
  author={Wang, Yuxi and Wang, Yikang and Zhang, Qishan and Nishizaki, Hiromitsu and Li, Ming},
  booktitle={Interspeech},
  year={2025}
}
```

---

## Repository Structure

```
VCapAV/
├── _engineering/          # Internal tools and engineering utilities
├── aasist/                # AASIST model and dataset-specific code
│   ├── config/            # AASIST config files (.conf)
│   ├── dev/               # Dev set data
│   ├── train/             # Training set data
│   ├── protocols/         # Protocol definition files
│   ├── models/            # Pretrained weights or model checkpoints
│   ├── utils.py           # AASIST-specific utility functions
│   └── download_dataset.py# Script to download dataset
├── clipclap/              # CLAP-based audio-visual feature extractor
├── configs/               # Training configs (ResNet, LCNN, etc.)
├── dataset/               # Data loading and preprocessing logic
├── exp/                   # Training output directory (e.g., checkpoints, logs)
├── log/                   # Logging and evaluation output
├── modules/               # Model architectures and backbones
├── utils/                 # Shared utility functions
├── features.py            # CLAP feature extraction interface
├── infer_from_score.py    # Score post-processing and metric calculation
├── main.py                # Entry point for training and evaluation
├── model_utils.py         # Model loading / saving / optimizer utils
├── other_utils.py         # Miscellaneous utility functions
├── requirement.txt        # Top-level Python package requirements
├── LICENSE
└── README.md
```

---

## Setup

### Environment

```bash
conda create -n vcapav python=3.8
conda activate vcapav
pip install -r requirement.txt
```

---

## Data Preparation

Organize the AASIST data directory as follows:

```
aasist/
├── train/
├── dev/
└── protocols/
```

### Protocol Composition

| Subset      | Composition                                                     | Total Samples        |
|-------------|------------------------------------------------------------------|----------------------|
| Audio Train | AudioLDM1, AudioLDM2, V2A-Mapper, Bonafide                      | 44,596 (11,149 × 4)  |
| Dev 1       | AudioLDM1, AudioLDM2, V2A-Mapper, Bonafide                      | 15,096 (3,774 × 4)   |
| Dev 2       | V2A-MLP, Audiocraft, Bonafide                                   | 11,322 (3,774 × 3)   |
| Dev 3       | All six types                                                   | 22,644 (3,774 × 6)   |
| Video Train | Bonafide, Kling                                                 | 11,192, 181          |
| Video Dev   | Bonafide, Kling                                                 | 3,731, 61            |

---

## Model Training

### AASIST

```bash
cd aasist
python main.py --config config/AASIST.conf --output_dir exp_result/
```

### ResNet

```bash
python main.py --config configs/resnet.yaml --output_dir exp_result/
```

### LCNN

```bash
python main.py --config configs/lcnn.yaml --output_dir exp_result/
```

---

## Evaluation

```bash
python main.py --config configs/resnet.yaml --eval
```

Optional post-processing:

```bash
python infer_from_score.py --score_path results/resnet_score.txt
```

---

## Pretrained Models

| Model                | Size   | Location                              |
|----------------------|--------|----------------------------------------|
| `AASIST.pth`         | 47 MB  | `aasist/models/weights/AASIST.pth`     |
| `AASIST-L.pth`       | 86 MB  | `aasist/models/weights/`               |
| `clap_htsat_tiny.pt` | 1.7 GB | (Download separately, see below)       |

> Note: Due to GitHub's file size limit, `clap_htsat_tiny.pt` is not included.  
> Please manually place it at: `clipclap/clap_htsat_tiny.pt`

---

## Benchmark Results

| Subset | Metric       | LightCNN | ResNet18 | AASIST |
|--------|--------------|----------|----------|--------|
| Dev 1  | EER (%)      | 0.8435   | 0.3135   | 0.02   |
|        | minDCF (%)   | 9.72     | 5.43     | 0.01   |
|        | AUC (%)      | 99.96    | 99.96    | 100    |
|        | Accuracy (%) | 98.42    | 99.7     | 99.92  |
| Dev 2  | EER (%)      | 18.89    | 14.67    | 2.96   |
|        | minDCF (%)   | 83.29    | 78.34    | 0.54   |
|        | AUC (%)      | 90.46    | 93.82    | 88.91  |
|        | Accuracy (%) | 73.52    | 75.17    | 99.52  |
| Dev 3  | EER (%)      | 9.67     | 9.82     | 1.78   |
|        | minDCF (%)   | 78.53    | 78.27    | 0.36   |
|        | AUC (%)      | 96.06    | 96.16    | 99.81  |
|        | Accuracy (%) | 88.24    | 87.87    | 96.29  |

---

## Contact

For questions or collaborations, please contact:  
**wa0009xi@e.ntu.edu.sg**
