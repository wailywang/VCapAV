# VCapAV

This repository provides the **VCapAV dataset**, which contains both forged and real audio-visual samples, with a focus on environmental sound manipulations. It includes baseline detection models such as **AASIST**, **ResNet18**, and **LCNN**, enabling focused evaluation of non-speech audio forgeries generated through Text-to-Audio and Video-to-Audio synthesis.

## Dataset Overview

<img src="configs/statistics.png" alt="Protocol Overview" width="70%"/>

For most audio-visual detection applications, we recommend starting with the `ResNet18 + LCNN` ensemble.

> **Note:** The complete VCapAV dataset, including all audio-visual samples, can be downloaded from Zenodo at:  
> **[https://doi.org/10.5281/zenodo.15498946](https://doi.org/10.5281/zenodo.15498946)**

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
> **Note:** Read the full paper describing the dataset and methodology at:  
> **[VCapAV: A Video-Caption-Based Audio-Visual Deepfake Detection Dataset (PDF)](https://sites.duke.edu/dkusmiip/files/2025/05/VCapAV-A-Video-Caption-Based-Audio-Visual-Deepfake-Detection-Dataset.pdf)**

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
<img src="configs/data_partitioning.png" alt="Protocol Overview" width="70%"/>

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

## Pretrained Models

| Model                | Size   | Location                                                                 |
|----------------------|--------|--------------------------------------------------------------------------|
| `AASIST.pth`         | 47 MB  | `aasist/models/weights/AASIST.pth`                                      |
| `AASIST-L.pth`       | 86 MB  | `aasist/models/weights/AASIST-L.pth`                                    |
| `clap_htsat_tiny.pt` | 1.7 GB | [`clipclap/clap_htsat_tiny.pt`](https://huggingface.co/mali6/autocap/blob/main/clap_htsat_tiny.pt) |

---

<div align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge&logo=open-source-initiative" alt="MIT License">
  </a> &ensp;
  <a href="mailto:wa0009xi@e.ntu.edu.sg">
    <img src="https://img.shields.io/badge/Contact-Email-blue?style=for-the-badge&logo=gmail" alt="Contact Email">
  </a>
</div>



