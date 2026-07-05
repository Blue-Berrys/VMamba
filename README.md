<div align="center">
<h1>IG-PaSCL</h1>
<h3>Illumination-Guided Penumbra Confidence Learning for Shadow Detection</h3>

[Paper](#citation) | [Code](https://github.com/Blue-Berrys/VMamba-Shadow/tree/penumbra-confidence) | [Results](#main-results) | [Getting Started](#getting-started) | [Model Zoo](#model-zoo)

</div>

## Updates

- **2026-07-05**: Repository README updated for the IG-PaSCL shadow detection project.
- **2026-07-05**: Paper draft reports SBU, ISTD, boundary-band, dark-distractor, penumbra-map, and direct-transfer evaluations.
- **Coming soon**: Cleaned training configs, checkpoints, and anonymous-review artifacts will be organized in the release branch.

## Abstract

Shadow detection is usually supervised as binary segmentation, although real shadows change continuously across umbra, penumbra, and weak-shadow regions. This hard supervision gives the same label to sharp and soft shadows and provides no target for the transition band where many errors remain.

IG-PaSCL keeps the standard binary shadow mask as the inference output, while adding a training-time penumbra confidence branch. The method first uses Brightness-Guided Scene Illumination Reference (BG-SIR) to reduce shadow contamination when estimating the scene illumination reference. It then converts binary masks into continuous penumbra supervision with a signed-distance transform and illumination cues. A lightweight confidence head learns this soft target from fused BG-SIR features and shares boundary-aware representations with the binary mask decoder.

On SBU, IG-PaSCL reaches 2.73 BER, improves boundary-band BER from 15.06 to 13.24, and reduces dark-distractor FPR from 7.69 to 6.72. On ISTD, it reduces BER from 1.33 to 1.28. The gains are concentrated in soft shadow boundaries and dark non-shadow distractors, which are difficult to diagnose with global BER alone.

## Overview

IG-PaSCL is built on the VMamba segmentation codebase and follows the standard MMSegmentation training workflow.

The current paper version contains two main components:

- **BG-SIR**: estimates a brightness-guided scene illumination reference and computes more stable shadow-to-reference contrast.
- **Penumbra Confidence Learning**: generates continuous soft-shadow targets from binary masks and trains an auxiliary confidence head for soft boundaries.

During inference, the official output remains a binary shadow mask. The penumbra map is used as training supervision and as an analysis signal for boundary and dark-distractor behavior.

## Main Results

### Standard Shadow Detection

| Dataset | Protocol | BER ↓ | FPR ↓ | FNR ↓ | F1 ↑ |
| :-- | :-- | --: | --: | --: | --: |
| SBU | train -> test | 2.73 | 2.77 | 2.70 | 93.24 |
| ISTD | train -> test | 1.28 | 1.47 | 1.10 | 95.68 |
| UCF | SBU -> UCF direct transfer | 6.89 | 6.23 | 7.54 | 82.38 |

### Mechanism-Oriented SBU Slices

| Method | Boundary BER ↓ | Boundary F1 ↑ | Dark FPR ↓ | Penumbra Pearson ↑ |
| :-- | --: | --: | --: | --: |
| Matched binary control | 15.06 | 85.14 | 7.69 | - |
| IG-PaSCL | 13.24 | 86.61 | 6.72 | 0.84 |

## Getting Started

### Installation

```bash
git clone git@github.com:Blue-Berrys/VMamba-Shadow.git
cd VMamba-Shadow
git checkout penumbra-confidence

conda create -n ig-pascl python=3.10 -y
conda activate ig-pascl

pip install -r requirements.txt
cd kernels/selective_scan
pip install .
cd ../..

pip install mmengine==0.10.1 mmcv==2.1.0 opencv-python-headless ftfy regex
pip install mmdet==3.3.0 mmsegmentation==1.2.2 mmpretrain==1.2.0
```

### Data Layout

Download the public shadow detection datasets from their project pages and place or symlink them under the `data/` directory. The datasets are not redistributed in this repository.

| Dataset | Public link | Expected local path |
| :-- | :-- | :-- |
| SBU Shadow | [SBU shadow dataset](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html) | `data/SBU-shadow/` |
| ISTD | [ISTD dataset / ST-CGAN project](https://github.com/DeepInsight-PCALab/ST-CGAN) | `data/ISTD_Dataset/` or `data/ISTD_binary/` |
| UCF Shadow | [UCF benchmark description](https://arxiv.org/abs/1810.05778) | `data/UCF/` |

```text
data/
  SBU-shadow/
    SBUTrain4KRecoveredSmall/
      ShadowImages/
      ShadowMasks/
    SBU-Test/
      ShadowImages/
      ShadowMasks/
  ISTD_Dataset/              # or ISTD_binary/, matching the config you run
    train/
      img/
      mask/
    test/
      img/
      mask/
  UCF/
    test/
      img/
      mask/
```

The exact dataset adapters follow the MMSegmentation dataset configuration files under `segmentation/configs/_base_/datasets/` and the paper-specific configs under `segmentation/configs/sbu/`.

### Training

```bash
cd segmentation

# SBU training
python tools/train.py \
  configs/sbu/shadow_icssm_penumbra.py \
  --work-dir work_dirs/ig_pascl_sbu

# ISTD fine-tuning
python tools/train.py \
  configs/sbu/shadow_icssm_istd.py \
  --work-dir work_dirs/ig_pascl_istd
```

For multi-GPU training:

```bash
cd segmentation
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  tools/train.py configs/sbu/shadow_icssm_penumbra.py \
  --work-dir work_dirs/ig_pascl_sbu_ddp \
  --launcher pytorch
```

### Evaluation

```bash
cd segmentation

# Standard SBU evaluation
python tools/test.py \
  configs/sbu/shadow_icssm_penumbra.py \
  work_dirs/ig_pascl_sbu/best_BER.pth

# Cross-dataset evaluation
python configs/sbu/test_cross_dataset.py

# Penumbra-map and hard-slice evaluation
python tools/evaluate_penumbra_maps.py \
  --config configs/sbu/shadow_icssm_penumbra.py \
  --checkpoint work_dirs/ig_pascl_sbu/best_BER.pth
```

## Model Zoo

| Model | Dataset | BER ↓ | Checkpoint |
| :-- | :-- | --: | :-- |
| IG-PaSCL | SBU | 2.73 | Coming soon |
| IG-PaSCL | ISTD | 1.28 | Coming soon |

## Repository Notes

This repository is a research fork built on VMamba and MMSegmentation. The paper-specific files are organized around the shadow detection segmentation workflow:

```text
segmentation/
  configs/sbu/                 # SBU, ISTD, and ablation configs
  tools/train.py               # MMSegmentation training entry
  tools/test.py                # MMSegmentation evaluation entry
  sbu_dataset.py               # SBU shadow dataset adapter
  ber_metric.py                # BER metric
```

Large datasets and checkpoints are not stored in Git. Use `work_dirs/` for local training outputs and keep pretrained weights under `pretrained/`.

## Citation

If this work is useful for your research, please cite:

```bibtex
@inproceedings{igpascl2027,
  title     = {Illumination-Guided Penumbra Confidence Learning for Shadow Detection},
  author    = {Anonymous},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2027}
}
```

This project is built on VMamba. Please also cite the original VMamba paper when using the backbone code.

## Acknowledgement

We thank the authors of VMamba, MMSegmentation, and the public SBU, ISTD, and UCF shadow detection datasets for their released code and benchmarks.
