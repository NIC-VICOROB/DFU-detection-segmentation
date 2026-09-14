"""
Zero-shot SAM inference (no LoRA fine-tuning), prompted with detection bounding
boxes. This is the baseline used to measure the gain from LoRA fine-tuning in
inference_sam_lora.py.

Uses Meta's official segment-anything package:
    pip install git+https://github.com/facebookresearch/segment-anything.git

Usage (from the repository root):
    python -m sam.inference_sam \
        --annotations sam/annotations_test.json \
        --split test \
        --images_dir data/DFUC2022/images/test \
        --sam_checkpoint weights/sam/sam_vit_b_01ec64.pth \
        --model_type vit_b \
        --output_dir predictions/nnunet_sam_zeroshot
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

try:
    from segment_anything import SamPredictor, sam_model_registry
except ImportError as e:
    raise ImportError(
        "The 'segment-anything' package is required for zero-shot inference. Install it with: "
        "pip install git+https://github.com/facebookresearch/segment-anything.git"
    ) from e


def run_inference(predictor: "SamPredictor", image_path: str, bbox: list) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    predictor.set_image(np.array(image))

    masks, _, _ = predictor.predict(box=np.array(bbox), multimask_output=False)

    return masks[0].astype(np.uint8) * 255


def main(annotations_path, split, images_dir, sam_checkpoint, model_type, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    with open(annotations_path) as f:
        annotations = json.load(f)[split]

    os.makedirs(output_dir, exist_ok=True)

    for filename, entry in annotations.items():
        bbox = entry["bbox"]
        if bbox == [0, 0, 0, 0]:
            print(f"[SKIP] No detection for {filename}.")
            continue

        image_path = os.path.join(images_dir, filename)
        if not os.path.exists(image_path):
            print(f"[WARN] Image not found: {image_path}")
            continue

        pred_mask = run_inference(predictor, image_path, bbox)
        Image.fromarray(pred_mask).save(os.path.join(output_dir, filename))

    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, required=True, help="JSON produced by create_annotations.py")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, default="weights/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--model_type", type=str, default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    main(
        annotations_path=args.annotations,
        split=args.split,
        images_dir=args.images_dir,
        sam_checkpoint=args.sam_checkpoint,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )
