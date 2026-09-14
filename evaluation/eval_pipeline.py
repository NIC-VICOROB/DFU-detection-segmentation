"""
End-to-end statistical comparison of several segmentation pipeline outputs (e.g. the
OR / AND / priority ensemble variants from segmentation/ensemble.py, or different
detection-guided-seg configurations). Computes per-image Dice for each prediction
set, aligns on the images common to all of them, then runs a pairwise Wilcoxon
signed-rank test between every pair of prediction sets.

Usage:
    python -m evaluation.eval_pipeline \
        --gt_dir data/DFUC/masks/test \
        --pred_dirs priority=predictions/ensemble_priority/test or=predictions/ensemble_or/test and=predictions/ensemble_and/test \
        --alpha 0.05 \
        --output_csv results/eval/ensemble_comparison.csv
"""
import argparse
import itertools
import os

import cv2
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.metrics import dice_coefficient


def compute_dice_per_image(pred_dir: str, gt_dir: str) -> dict:
    """Returns {filename_stem: dice}."""
    results = {}
    for gt_fname in os.listdir(gt_dir):
        if not gt_fname.endswith(".png"):
            continue

        pred_path = os.path.join(pred_dir, gt_fname)
        if not os.path.exists(pred_path):
            continue

        gt = (cv2.imread(os.path.join(gt_dir, gt_fname), cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        pred = (cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        results[os.path.splitext(gt_fname)[0]] = dice_coefficient(gt, pred)

    return results


def parse_pred_dirs(pred_dirs_arg: list) -> dict:
    """Parses ["name=path", ...] into {name: path}."""
    parsed = {}
    for entry in pred_dirs_arg:
        name, path = entry.split("=", 1)
        parsed[name] = path
    return parsed


def main(gt_dir: str, pred_dirs: dict, alpha: float, output_csv: str):
    dice_per_set = {name: compute_dice_per_image(path, gt_dir) for name, path in pred_dirs.items()}

    common_files = set.intersection(*[set(d.keys()) for d in dice_per_set.values()])
    common_files = sorted(common_files)
    print(f"{len(common_files)} images common to all {len(pred_dirs)} prediction sets.")

    print("\n=== MEAN DICE ===")
    for name, dice_dict in dice_per_set.items():
        values = np.array([dice_dict[f] for f in common_files])
        print(f"{name}: mean={values.mean():.4f}  std={values.std():.4f}")

    print("\n=== PAIRWISE WILCOXON (per-image Dice) ===")
    rows = []
    for name_a, name_b in itertools.combinations(dice_per_set.keys(), 2):
        values_a = np.array([dice_per_set[name_a][f] for f in common_files])
        values_b = np.array([dice_per_set[name_b][f] for f in common_files])

        stat, p_value = wilcoxon(values_a, values_b)
        significant = p_value < alpha
        print(f"{name_a} vs {name_b}: W={stat:.2f}, p={p_value:.6f} ({'SIGNIFICANT' if significant else 'NOT SIGNIFICANT'})")

        rows.append({
            "comparison": f"{name_a}_vs_{name_b}",
            "mean_dice_a": values_a.mean(),
            "mean_dice_b": values_b.mean(),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "significant": significant,
        })

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"\nComparison table saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--pred_dirs", type=str, nargs="+", required=True, help="One or more name=path pairs, e.g. priority=predictions/ensemble_priority/test")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    main(gt_dir=args.gt_dir, pred_dirs=parse_pred_dirs(args.pred_dirs), alpha=args.alpha, output_csv=args.output_csv)