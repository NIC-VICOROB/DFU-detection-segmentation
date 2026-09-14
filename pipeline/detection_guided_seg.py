"""
Detection-guided SAM segmentation pipeline.

Prompts SAM with a bounding box "guide" that can come from either:
  - a detector's CSV output (filename, xmin, ymin, xmax, ymax), e.g. YOLO / Faster
    R-CNN / nnDetection / Grounding DINO / an ensemble (detection/ensemble_wbf.py), or
  - a segmentation model's predicted masks (e.g. nnUNet, YOLO-seg), from which a
    bounding box is derived per image with get_bounding_box.

Usage (from the repository root):
    # Guided by a detector's CSV, zero-shot SAM
    python -m pipeline.detection_guided_seg \
        --bbox_csv results/yolo_predictions.csv \
        --images_dir data/DFUC/images/test \
        --sam_variant zero_shot --sam_checkpoint weights/sam/sam_vit_b_01ec64.pth \
        --output_dir predictions/yolo_sam/test

    # Guided by nnUNet's predicted masks, LoRA fine-tuned SAM
    python -m pipeline.detection_guided_seg \
        --mask_dir results/nnunet/dfuc/postprocessed \
        --images_dir data/DFUC/images/test \
        --sam_variant lora --lora_weights weights/sam_lora/lora_rank512.safetensors --rank 512 \
        --output_dir predictions/nn_sam/test
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image

from src.utils import get_bounding_box

THIRD_PARTY_DIR = os.environ.get(
    "SAM_LORA_REPO",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam", "third_party", "Sam_LoRA"),
)


def load_bbox_dict_from_csv(csv_path: str) -> dict:
    """Detector guide: one box per image, keyed by filename stem (union of all detections)."""
    df = pd.read_csv(csv_path)
    bbox_dict = {}
    for fname, group in df.groupby("filename"):
        stem = os.path.splitext(str(fname))[0]
        bbox_dict[stem] = [
            int(group["xmin"].min()),
            int(group["ymin"].min()),
            int(group["xmax"].max()),
            int(group["ymax"].max()),
        ]
    return bbox_dict


def load_bbox_dict_from_masks(mask_dir: str, mask_ext: str = ".png") -> dict:
    """Segmentation-model guide: derive a box from each predicted mask's extent."""
    bbox_dict = {}
    for fname in sorted(os.listdir(mask_dir)):
        if not fname.endswith(mask_ext):
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, fname)).convert("L"))
        stem = os.path.splitext(fname)[0]
        bbox_dict[stem] = get_bounding_box((mask > 0).astype(np.uint8))
    return bbox_dict


def build_predictor(sam_variant: str, sam_checkpoint: str, model_type: str, lora_weights: str, rank: int):
    if sam_variant == "zero_shot":
        from segment_anything import SamPredictor, sam_model_registry

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    else:
        if THIRD_PARTY_DIR not in sys.path:
            sys.path.insert(0, THIRD_PARTY_DIR)
        try:
            from src.lora import LoRA_sam
            from src.segment_anything import SamPredictor, build_sam_vit_b
        except ImportError as e:
            raise ImportError(
                f"Could not import the Sam_LoRA dependency from '{THIRD_PARTY_DIR}'. "
                "Clone https://github.com/MathieuNlp/Sam_LoRA there (or set SAM_LORA_REPO). "
                "See sam/README.md."
            ) from e

        sam = build_sam_vit_b(checkpoint=sam_checkpoint)
        sam_lora = LoRA_sam(sam, rank)
        sam_lora.load_lora_parameters(lora_weights)
        sam = sam_lora.sam

    import torch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def run_inference(predictor, image_path: str, bbox: list) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    predictor.set_image(np.array(image))
    masks, _, _ = predictor.predict(box=np.array(bbox), multimask_output=False)
    return masks[0].astype(np.uint8) * 255


def main(args):
    if args.bbox_csv:
        bbox_dict = load_bbox_dict_from_csv(args.bbox_csv)
    else:
        bbox_dict = load_bbox_dict_from_masks(args.mask_dir, args.mask_ext)

    predictor = build_predictor(args.sam_variant, args.sam_checkpoint, args.model_type, args.lora_weights, args.rank)

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = sorted(
        f for f in os.listdir(args.images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for filename in image_files:
        stem = os.path.splitext(filename)[0]
        bbox = bbox_dict.get(stem, [0, 0, 0, 0])
        if bbox == [0, 0, 0, 0]:
            print(f"[SKIP] No guide bbox for {filename}.")
            continue

        image_path = os.path.join(args.images_dir, filename)
        pred_mask = run_inference(predictor, image_path, bbox)
        Image.fromarray(pred_mask).save(os.path.join(args.output_dir, f"{stem}.png"))

    print(f"Saved predictions to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    guide_group = parser.add_mutually_exclusive_group(required=True)
    guide_group.add_argument("--bbox_csv", type=str, help="Detector CSV guide (filename, xmin, ymin, xmax, ymax)")
    guide_group.add_argument("--mask_dir", type=str, help="Segmentation model guide: folder of predicted masks")
    parser.add_argument("--mask_ext", type=str, default=".png")

    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--sam_variant", type=str, required=True, choices=["zero_shot", "lora"])
    parser.add_argument("--sam_checkpoint", type=str, default="weights/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="Only used for --sam_variant zero_shot")
    parser.add_argument("--lora_weights", type=str, default=None, help="Required for --sam_variant lora")
    parser.add_argument("--rank", type=int, default=512, help="Only used for --sam_variant lora")

    args = parser.parse_args()
    if args.sam_variant == "lora" and not args.lora_weights:
        parser.error("--lora_weights is required when --sam_variant lora")

    main(args)