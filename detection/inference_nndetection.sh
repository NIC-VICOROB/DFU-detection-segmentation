#!/bin/bash
# nnDetection inference for DFU detection

# ─────────────────────────────────────────
# Step 1: Convert test images to pseudo-3D NIfTI format
# ─────────────────────────────────────────
python detection/convert_to_nndetection.py \
    --images_dir data/DFUC/images/test \
    --labels_dir data/DFUC/labels_detection \
    --output_dir data/nndetection/Task100_DFU \
    --z_depth 96

# ─────────────────────────────────────────
# Step 2: Predict
# ─────────────────────────────────────────
docker run --gpus all \
    -v ${det_data}:/opt/data \
    -v ${det_models}:/opt/models \
    --shm-size=24gb \
    nndetection:0.1 \
    nndet_predict 100 RetinaUNetV001_D3V001_3d --fold -1

# ─────────────────────────────────────────
# Step 3: Evaluate
# ─────────────────────────────────────────
docker run --gpus all \
    -v ${det_data}:/opt/data \
    -v ${det_models}:/opt/models \
    --shm-size=24gb \
    nndetection:0.1 \
    nndet_eval 100 RetinaUNetV001_D3V001_3d -1 --test --boxes --analyze_boxes