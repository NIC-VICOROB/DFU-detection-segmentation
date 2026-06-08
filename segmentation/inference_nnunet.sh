#!/bin/bash
# nnUNet inference for DFU segmentation

export nnUNet_raw="data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="data/nnunet/nnUNet_preprocessed"
export nnUNet_results="weights/nnunet"

POSTPROCESSING_PKL="weights/nnunet/Dataset001_DFU/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl"
PLANS_JSON="weights/nnunet/Dataset001_DFU/nnUNetTrainer__nnUNetPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json"

# ─────────────────────────────────────────
# DFUC test set
# ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -d Dataset001_DFU \
    -i data/nnunet/nnUNet_raw/Dataset001_DFU/imagesTs \
    -o results/nnunet/dfuc/raw \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer -c 2d -p nnUNetPlans

nnUNetv2_apply_postprocessing \
    -i results/nnunet/dfuc/raw \
    -o results/nnunet/dfuc/postprocessed \
    -pp_pkl_file $POSTPROCESSING_PKL \
    -np 8 \
    -plans_json $PLANS_JSON

# ─────────────────────────────────────────
# FUSeg test set (out-of-domain evaluation)
# ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 nnUNetv2_predict \
    -d Dataset001_DFU \
    -i data/nnunet/nnUNet_raw/Dataset002_FUSeg/imagesTs \
    -o results/nnunet/fuseg/raw \
    -f 0 1 2 3 4 \
    -tr nnUNetTrainer -c 2d -p nnUNetPlans

nnUNetv2_apply_postprocessing \
    -i results/nnunet/fuseg/raw \
    -o results/nnunet/fuseg/postprocessed \
    -pp_pkl_file $POSTPROCESSING_PKL \
    -np 8 \
    -plans_json $PLANS_JSON