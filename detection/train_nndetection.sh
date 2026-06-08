#!/bin/bash
# nnDetection training for DFU detection
# Note: nnDetection is a 3D framework. Since DFU images are 2D,
# we construct pseudo-3D volumes by replicating the 2D slice along Z.
# Z=96 was chosen as it is divisible by 16 (4 downsampling levels).
# This is a known limitation discussed in the paper.

# ─────────────────────────────────────────
# Step 1: Convert 2D images to pseudo-3D NIfTI format
# ─────────────────────────────────────────
python detection/convert_to_nndetection.py \
    --images_dir data/DFUC/images/train \
    --labels_dir data/DFUC/labels_detection \
    --output_dir data/nndetection/Task100_DFU \
    --z_depth 96

# ─────────────────────────────────────────
# Step 2: Build Docker image
# ─────────────────────────────────────────
docker build -t nndetection:0.1 \
    --build-arg env_det_num_threads=6 \
    --build-arg env_det_verbose=1 .

# ─────────────────────────────────────────
# Step 3: Preprocess
# ─────────────────────────────────────────
docker run --gpus all \
    -v ${det_data}:/opt/data \
    -v ${det_models}:/opt/models \
    --shm-size=24gb \
    nndetection:0.1 \
    nndet_prep 100

# ─────────────────────────────────────────
# Step 4: Unpack
# ─────────────────────────────────────────
docker run --gpus all \
    -v ${det_data}:/opt/data \
    -v ${det_models}:/opt/models \
    --shm-size=24gb \
    nndetection:0.1 \
    nndet_unpack /opt/data/Task100_DFU/preprocessed/D3V001_3d/imagesTr 6

# ─────────────────────────────────────────
# Step 5: Train (5 folds)
# ─────────────────────────────────────────
for fold in 0 1 2 3 4; do
    docker run --gpus all \
        -v ${det_data}:/opt/data \
        -v ${det_models}:/opt/models \
        --shm-size=24gb \
        nndetection:0.1 \
        nndet_train 100 --sweep -o exp.fold=${fold}
done

# ─────────────────────────────────────────
# Step 6: Consolidate
# ─────────────────────────────────────────
docker run --gpus all \
    -v ${det_data}:/opt/data \
    -v ${det_models}:/opt/models \
    --shm-size=24gb \
    nndetection:0.1 \
    nndet_consolidate 100 RetinaUNetV001_D3V001_3d --sweep_boxes