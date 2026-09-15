# Weights

Model checkpoints are not tracked in this repository (see `.gitignore`). Place
downloaded or trained weights here so the scripts' default paths work as-is, or pass
`--weights_path` / `--checkpoint` explicitly to use a different location.

| Path | Source |
|---|---|
| `weights/sam/sam_vit_b_01ec64.pth` | [Meta SAM ViT-B checkpoint](https://github.com/facebookresearch/segment-anything#model-checkpoints) |
| `weights/sam_lora/` | Produced by `sam/train_lora.py` |
| `weights/yolo_det/`, `weights/yolo_seg/` | Produced by `detection/train_yolo.py`, `segmentation/train_yolo_seg.py` |
| `weights/faster_rcnn/` | Produced by `detection/train_faster_rcnn.py` |
| `weights/mask_rcnn/` | Produced by `segmentation/train_maskrcnn.sh` |
| `weights/segnet/` | Produced by `segmentation/train_segnet.py` |
| `weights/nnunet/`, nnDetection models | Produced by the Docker-based training scripts (`segmentation/train_nnunet.sh`, `detection/train_nndetection.sh`) |
