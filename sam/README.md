# SAM + LoRA

Detection-guided SAM segmentation: bounding boxes from a detector (or ensemble) are
used to prompt SAM, optionally fine-tuned with LoRA on the ViT-B image encoder.

## Setup

1. **Zero-shot baseline** (`inference_sam.py`) only needs Meta's official package:
   ```
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

2. **LoRA fine-tuning and inference** (`train_lora.py`, `inference_sam_lora.py`) depend on
   [MathieuNlp/Sam_LoRA](https://github.com/MathieuNlp/Sam_LoRA), which is not vendored in
   this repository (same pattern as the nnUNet/nnDetection Docker dependencies elsewhere
   in this repo). Clone it and install its requirements:
   ```
   git clone https://github.com/MathieuNlp/Sam_LoRA.git sam/third_party/Sam_LoRA
   pip install -r sam/third_party/Sam_LoRA/requirements.txt  # or via poetry, see their README
   pip install monai
   ```
   If cloned elsewhere, point to it with `SAM_LORA_REPO=/path/to/Sam_LoRA`.

3. Download the SAM ViT-B checkpoint (`sam_vit_b_01ec64.pth`) into `weights/sam/`.

## Pipeline

1. `create_annotations.py` — builds a JSON pairing each mask filename with a detection
   bounding box (from a detector's or ensemble's CSV output).
2. `train_lora.py` — fine-tunes SAM's image encoder with LoRA, prompted with the boxes
   from step 1 (config in `lora_config.yaml`).
3. `inference_sam.py` / `inference_sam_lora.py` — run zero-shot or LoRA-fine-tuned SAM
   inference on the annotations from step 1.

All scripts are run as modules from the repository root, e.g.:
```
python -m sam.create_annotations --bbox_csv ... --mask_dir ... --output_json sam/annotations_test.json
python -m sam.train_lora --config sam/lora_config.yaml
python -m sam.inference_sam_lora --annotations sam/annotations_test.json --split test --images_dir ... --lora_weights ...
```
