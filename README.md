# VCapAV

**VCapAV** is an audio-visual deepfake detection benchmark and toolkit, featuring three key models: **AASIST**, **ResNet18**, and **LCNN**. The benchmark supports both unimodal and cross-modal forgery detection using spatial frequency features, log-Fbank spectrograms, and raw waveform encodings.

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

For most audio-visual detection applications, we recommend starting with the `ResNet18 + LCNN` ensemble.

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
├── configs/               # Training configs (ResNet18, LCNN, etc.)
├── dataset/               # Data loading and preprocessing logic
├── exp/                   # Training output directory
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

## Data Preparation

Organize dataset directory as follows:

```
dataset/
├── train/
│   ├── wav.scp
│   └── utt2label
├── dev/
│   ├── wav.scp
│   └── utt2label
```

### File Format Details

**`wav.scp`**  
Each line should contain an utterance ID and the absolute path to the corresponding audio file:

```
UtteranceID1 /path/to/audio1.wav
UtteranceID2 /path/to/audio2.wav
```

**`utt2label`**  
Each line should contain an utterance ID and its label (`bonafide` or `spoof`). This file is required for both training and development sets:

```
UtteranceID1 bonafide
UtteranceID2 spoof
```
### Label Mapping

The script automatically maps the string labels to integers:

- `bonafide` or `genuine` → `1`
- `spoof` or `fake` → `0`

Make sure all utterance IDs in `utt2label` exist in `wav.scp`.

---

## Model Training

### ResNet18

```bash
for seed in 1; do
  python3 -u main.py --comment "train_clean_offset_Resnet" --track "FAD" \
      --gpu 1 --workers 6 --batch_size 64 \
      --exp_dir "exp/clean" --log_dir "log/clean" \
      --trn_data_name "train" --dur_range 7 7 \
      --dev_data_name "dev" \
      --feat "logFbankCal" --preemph False --vad False --data_aug False --snr_range 0 20 \
      --aug_rate 0.7 --is_specaug False --speed_aug False --reverb False \
      --model ResNet18_ASP --model_cfg "configs/main_80.yaml" \
      --mse_loss False \
      --model_pretrain None --loss ce --model_freeze_epoch 0 \
      --classifier Linear --angular_m 0.2 --angular_s 32 --max_step 200000 --dropout 0.4 \
      --use_amp False --start_epoch 0 --num_epochs 150 --warm_up_epoch 1 --lr 0.001 \
      --seed ${seed} \
      --early_stop 50 --offset True
done
```

### LCNN

```bash
for seed in 1; do
  python3 -u main.py --comment "CleanTrain_oneinten_LCNN" --track "ASVspoof5" \
      --gpu 1 --workers 6 --batch_size 64 \
      --exp_dir "exp/clean" --log_dir "log/clean" \
      --trn_data_name "train" --dur_range 7 7 \
      --dev_data_name "dev" \
      --feat "logFbankCal" --preemph False --vad False --data_aug False --snr_range 0 20 \
      --aug_rate 0.7 --is_specaug False --speed_aug False --reverb False \
      --model LightCNN_lstm --model_cfg "configs/main_80.yaml" \
      --mse_loss False \
      --model_pretrain None --loss ce --model_freeze_epoch 0 \
      --classifier Linear --angular_m 0.2 --angular_s 32 --max_step 200000 --dropout 0.4 \
      --use_amp False --start_epoch 0 --num_epochs 150 --warm_up_epoch 1 --lr 0.001 \
      --seed ${seed} \
      --early_stop 50
done
```
### AASIST

```bash
cd aasist
python3 main.py --config ./config/AASIST.conf --comment "aasist"
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

| Model                | Size   | Location                                                                 |
|----------------------|--------|--------------------------------------------------------------------------|
| `AASIST.pth`         | 47 MB  | `aasist/models/weights/AASIST.pth`                                      |
| `AASIST-L.pth`       | 86 MB  | `aasist/models/weights/AASIST-L.pth`                                    |
| `clap_htsat_tiny.pt` | 1.7 GB | [`clipclap/clap_htsat_tiny.pt`](https://huggingface.co/mali6/autocap/blob/main/clap_htsat_tiny.pt) |

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
