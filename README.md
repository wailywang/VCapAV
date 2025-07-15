# VCapAV

[📄 Paper] [📦 Models] [📊 Results] [▶ Training Code]

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

| Model    | Dataset  | ACC (%) | AUC (%) | EER (%) |
|----------|----------|---------|---------|---------|
| AASIST   | Wan2.1   | 97.3    | 98.4    | 2.15    |
| ResNet18 | Kling    | 96.2    | 97.1    | 3.60    |
| LCNN     | CogVideo | 96.7    | 97.9    | 3.05    |

---

## Contact

Yuxi Wang (wa0009xi@e.ntu.edu.sg)
