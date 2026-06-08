#!/bin/bash
# Mask R-CNN training using MMDetection
# Training is handled via MMDetection's own train script.
# The config file needs to be modified before running (see configs/mask_rcnn_ulcer.py).

# Install MMDetection if needed:
# pip install -U openmim
# mim install mmengine mmcv mmdet

python mmdetection/tools/train.py configs/mask_rcnn_ulcer.py