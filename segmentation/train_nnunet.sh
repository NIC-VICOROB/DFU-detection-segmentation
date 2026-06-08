#!/bin/bash
# nnUNet training for DFU segmentation
# nnUNet is a self-configuring framework — training is done via command line.
# Before running, convert your data to nnUNet format (see data/README.md).

# ─────────────────────────────────────────
# Step 1: Set environment variables
# ─────────────────────────────────────────
export nnUNet_raw="data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="data/nnunet/nnUNet_preprocessed"
export nnUNet_results="weights/nnunet"

# ─────────────────────────────────────────
# Step 2: Plan and preprocess
# ─────────────────────────────────────────
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

# ─────────────────────────────────────────
# Step 3: Train 5 folds
# ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset001_DFU 2d 0 --npz
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset001_DFU 2d 1 --npz
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset001_DFU 2d 2 --npz
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset001_DFU 2d 3 --npz
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset001_DFU 2d 4 --npz

# ─────────────────────────────────────────
# Step 4: Find best configuration
# ─────────────────────────────────────────
nnUNetv2_find_best_configuration Dataset001_DFU
# Expected best: nnUNetTrainer__nnUNetPlans__2d with Dice 0.7686