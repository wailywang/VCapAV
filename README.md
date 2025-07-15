# VCapAV

[Paper] [Models] [Results] [Training Code]

**VCapAV** is an audio-visual deepfake detection benchmark and toolkit featuring three key models: AASIST, ResNet, and LCNN. The benchmark evaluates both unimodal and cross-modal forgeries using spatial frequency features, audio log-fbanks, and raw waveform encodings.

| Category                | Number of Clips   | Total Duration (hours)   |
|:------------------------|:------------------|:-------------------------|
| Real Video              | 14,923            | 41.45                    |
| Fake Video              | 242               | 0.67                     |
| Real Audio              | 14,923            | 41.45                    |
| Fake Audio              | 74,615            | 207.26                   |
| Real Video + Real Audio | 14,923            | 41.45                    |
| Real Video + Fake Audio | 74,615            | 207.26                   |
| Fake Video + Real Audio | 242               | 0.67                     |
| Fake Video + Fake Audio | 1,210             | 3.36                     |
| Total                   | **90,990**        | **252.75**               |


For most audio-visual detection applications, we recommend starting with the `ResNet + LCNN` ensemble.

> **Note:** Results are evaluated on the closed-world AV-deepfake task using the Wan2.1/Kling/CogVideo benchmark in `dev`.

---

## Paper

```
@inproceedings{wang2025vcapav,
  title={VCapAV: A Video-Caption Based Audio-Visual Deepfake Detection Dataset},
  author={Wang, Yuxi and Wang, Yikang and Zhang, Qishan and Nishizaki, Hiromitsu and Li, Ming},
  booktitle={Interspeech},
  year={2025}
}
```

## File Structure

```
VCapAV/
├── aasist/        # AASIST model code and config
├── clipclap/      # CLAP-based AV feature extractor
├── configs/       # Training configs for ResNet / LCNN
├── modules/       # Model architectures
├── dataset/       # Data handling
├── main.py        # Training entry point
├── features.py    # CLAP feature extractor
└── README.md
```

---

## Usage

### Installation

```bash
conda create -n vcapav python=3.8
conda activate vcapav
pip install -r requirement.txt
```

### Data Preparation

Ensure protocol files and data are organized as:

```
aasist/
├── train/
├── dev/
└── protocols/
```
---
| Subset      | Composition                                                     | Total Size          |
|:------------|:----------------------------------------------------------------|:--------------------|
| Audio Train | AudioLDM1, AudioLDM2, V2A-Mapper, Bonafide                      | 44,596 (11,149 * 4) |
| Dev 1       | AudioLDM1, AudioLDM2, V2A-Mapper, Bonafide                      | 15,096 (3,774 * 4)  |
| Dev 2       | V2A-MLP, Audiocraft, Bonafide                                   | 11,322 (3,774 * 3)  |
| Dev 3       | AudioLDM1, AudioLDM2, V2A-Mapper, V2A-MLP, Audiocraft, Bonafide | 22,644 (3,774 * 6)  |
| Video Train | Bonafide, Kling                                                 | 11,192, 181         |
| Video Dev   | Bonafide, Kling                                                 | 3,731, 61           |


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

or

```bash
python infer_from_score.py --score_path results/resnet_score.txt
```

---

## Pretrained Models

| Model File                | Size   | Path                                |
|--------------------------|--------|-------------------------------------|
| `AASIST.pth`             | 47 MB  | `aasist/models/weights/AASIST.pth` |
| `AASIST-L.pth`           | 86 MB  | `aasist/models/weights/`           |
| `clap_htsat_tiny.pt`     | 1.7 GB | 🔗 [Download manually](#)           |

⚠️ Due to GitHub's 100MB file size limit, `clap_htsat_tiny.pt` is **not included**.  
Please download and place it manually under `clipclap/clap_htsat_tiny.pt`.

---

## Benchmark Results

| Subset   | Metrics    |   LightCNN |   ResNet18 |   AASIST |
|:---------|:-----------|-----------:|-----------:|---------:|
| Dev 1    | EER (%)    |     0.8435 |     0.3135 |     0.02 |
| Dev 1    | minDCF (%) |     9.72   |     5.43   |     0.01 |
| Dev 1    | AUC (%)    |    99.96   |    99.96   |   100    |
| Dev 1    | Acc. (%)   |    98.42   |    99.7    |    99.92 |
| Dev 2    | EER (%)    |    18.8924 |    14.6728 |     2.96 |
| Dev 2    | minDCF (%) |    83.29   |    78.34   |     0.54 |
| Dev 2    | AUC (%)    |    90.46   |    93.82   |    88.91 |
| Dev 2    | Acc. (%)   |    73.52   |    75.17   |    99.52 |
| Dev 3    | EER (%)    |     9.6714 |     9.8198 |     1.78 |
| Dev 3    | minDCF (%) |    78.53   |    78.27   |     0.36 |
| Dev 3    | AUC (%)    |    96.06   |    96.16   |    99.81 |
| Dev 3    | Acc. (%)   |    88.24   |    87.87   |    96.29 |

---

## Contact

wa0009xi@e.ntu.edu.sg
