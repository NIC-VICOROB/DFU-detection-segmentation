"""
Per-image segmentation evaluation: Dice and IoU between predicted and groundtruth
binary masks, plus dataset-level mean/std.

Usage:
    python -m evaluation.eval_segmentation \
        --gt_dir data/DFUC/masks/test \
        --pred_dir predictions/nn_sam/test \
        --output_csv results/eval/nn_sam_test_dice.csv
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from src.metrics import dice_coefficient, mask_iou


def evaluate_per_image(gt_dir: str, pred_dir: str) -> pd.DataFrame:
    gt_files = {os.path.splitext(f)[0]: f for f in os.listdir(gt_dir) if f.endswith(".png")}
    pred_files = {os.path.splitext(f)[0]: f for f in os.listdir(pred_dir) if f.endswith(".png")}

    results = []
    for stem, gt_fname in gt_files.items():
        if stem not in pred_files:
            print(f"[WARN] No prediction found for {gt_fname}, skipping.")
            continue

        gt = (cv2.imread(os.path.join(gt_dir, gt_fname), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        pred = (cv2.imread(os.path.join(pred_dir, pred_files[stem]), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        results.append({
            "filename": gt_fname,
            "dice": dice_coefficient(gt, pred),
            "iou": mask_iou(gt, pred),
        })

    return pd.DataFrame(results)


def main(gt_dir: str, pred_dir: str, output_csv: str):
    per_image = evaluate_per_image(gt_dir, pred_dir)

    print(f"Evaluated {len(per_image)} images")
    print(f"Dice: mean={per_image['dice'].mean():.4f}  std={per_image['dice'].std():.4f}")
    print(f"IoU:  mean={per_image['iou'].mean():.4f}  std={per_image['iou'].std():.4f}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        per_image.to_csv(output_csv, index=False)
        print(f"Per-image metrics saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    main(gt_dir=args.gt_dir, pred_dir=args.pred_dir, output_csv=args.output_csv)