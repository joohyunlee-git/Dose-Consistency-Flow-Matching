# Dose-Consistency Flow Matching (DCFM)

Official implementation of **Dose-Consistency Flow Matching** for ultra-low-dose PET image enhancement.

This repository is built on top of Meta's Flow Matching library:
> https://github.com/facebookresearch/flow_matching

---

## Overview

DCFM trains a flow matching model conditioned on the dose reduction factor (DRF) via FiLM, with a consistency regularization loss that enforces dose-invariant predictions across independently sampled dose levels. A curriculum dose-gap schedule gradually increases the difficulty of sampled dose pairs during training.

Key features:
- Multi-dose training with a single model
- FiLM-based dose conditioning
- Consistency regularization between dose pairs
- Supports `fm`, `dm`, `unet`, and `gan` model types

---

## Requirements

```bash
pip install -r requirements.txt
```

For PyTorch, follow the [official installation guide](https://pytorch.org/get-started/locally/) for your CUDA version.

---

## Data

Input data should be in HDF5 (`.h5`) format with the following structure:

```
file.h5
├── thumbnail/
│   ├── img          # downsampled volume for patch sampling
│   └── attrs: chunk_size
├── 1/               # normal-dose (ND)
│   ├── img
│   └── attrs: scale, p9999
├── 20/              # DRF 20
│   ├── img
│   └── attrs: scale, p9999
└── 50/              # DRF 50
    ├── img
    └── attrs: scale, p9999
```

A splits JSON file specifying train/val/test file lists per fold is required:

```json
{
  "fold_0": {
    "train": ["path/to/file1.h5", ...],
    "val":   ["path/to/file2.h5", ...],
    "test":  ["path/to/file3.h5", ...]
  }
}
```

---

## Training

```bash
python train_dcfm.py \
    --dataset <dataset_name> \
    --model fm \
    --fm_pred_type velocity \
    --fm_path cond_ot \
    --dose_levels 50,20,10,4 \
    --nd 1 \
    --ref 50 \
    --data_path /path/to/data \
    --splits_json /path/to/splits.json \
    --train_dir /path/to/output \
    --duration 100000 \
    --batch_size 2 \
    --patch_size 64 \
    --cons_loss 0.3 \
    --cons_ramp_start 0.1 \
    --cons_ramp_end 0.5 \
    --gap_init 0.3 \
    --gap_ramp_start 0.0 \
    --gap_ramp_end 0.5
```

To resume training:

```bash
python train_dcfm.py \
    --resume /path/to/output/run_dir/checkpoint-XXXXX.pth \
    --duration 200000 \
    ...
```

---

## Inference

```bash
python test_dcfm.py \
    --dataset <dataset_name> \
    --model fm \
    --fm_pred_type velocity \
    --ld 50 \
    --nd 1 \
    --ref 50 \
    --nfe 10 \
    --ode_method euler \
    --data_path /path/to/data \
    --splits_json /path/to/splits.json \
    --test_dir /path/to/results \
    --resume /path/to/output/run_dir/checkpoint-XXXXX.pth
```

Results are saved as `.h5` files under `--test_dir`.

---

## Citation

If you find this code useful, please consider citing our work (citation info TBD).

---

## Acknowledgements

This project builds on the [flow-matching](https://github.com/facebookresearch/flow_matching) library by Meta Platforms, Inc., licensed under CC-by-NC.
