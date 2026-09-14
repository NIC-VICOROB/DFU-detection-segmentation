"""
Run inference with a LoRA fine-tuned SAM model, prompted with detection bounding
boxes (see create_annotations.py for the expected input JSON format).

Depends on https://github.com/MathieuNlp/Sam_LoRA, same as train_lora.py.

Usage (from the repository root):
    python -m sam.inference_sam_lora \
        --annotations sam/annotations_test.json \
        --split test \
        --images_dir data/DFUC2022/images/test \
        --lora_weights weights/sam_lora/lora_rank512.safetensors \
        --rank 512 \
        --output_dir predictions/nnunet_sam
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

THIRD_PARTY_DIR = os.environ.get(
    "SAM_LORA_REPO",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party", "Sam_LoRA"),
)
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

from sam.src import utils


def expand_bbox(bbox: list, img_width: int, img_height: int, scale: float = 0.0) -> list:
    """Optionally pad a bounding box before prompting SAM, clipped to the image bounds."""
    x1, y1, x2, y2 = bbox
    dx = int((x2 - x1) * scale)
    dy = int((y2 - y1) * scale)
    return [
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_width - 1, x2 + dx),
        min(img_height - 1, y2 + dy),
    ]


def run_inference(predictor: "SamPredictor", image_path: str, bbox: list, expand_scale: float) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    box = expand_bbox(bbox, img_w, img_h, scale=expand_scale)

    predictor.set_image(np.array(image))
    masks, _, _ = predictor.predict(box=np.array(box), multimask_output=False)

    return masks[0].astype(np.uint8) * 255


def main(annotations_path, split, images_dir, lora_weights, sam_checkpoint, rank, output_dir, expand_scale):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    sam_lora = LoRA_sam(sam, rank)
    sam_lora.load_lora_parameters(lora_weights)
    model = sam_lora.sam
    model.eval()
    model.to(device)
    predictor = SamPredictor(model)

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

        pred_mask = run_inference(predictor, image_path, bbox, expand_scale)
        Image.fromarray(pred_mask).save(os.path.join(output_dir, filename))

    print(f"Saved predictions to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=str, required=True, help="JSON produced by create_annotations.py")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--lora_weights", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, default="weights/sam/sam_vit_b_01ec64.pth")
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--expand_scale", type=float, default=0.0, help="Fractional bbox padding before prompting SAM")
    args = parser.parse_args()

    main(
        annotations_path=args.annotations,
        split=args.split,
        images_dir=args.images_dir,
        lora_weights=args.lora_weights,
        sam_checkpoint=args.sam_checkpoint,
        rank=args.rank,
        output_dir=args.output_dir,
        expand_scale=args.expand_scale,
    )
