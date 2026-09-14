"""
Per-image detection evaluation: matches predicted boxes to groundtruth boxes by IoU
and reports Precision, Recall and F1 per image, plus dataset-level means.

Usage:
    python -m evaluation.eval_detection \
        --gt_csv data/DFUC/annotations/groundtruth.csv \
        --pred_csv results/yolo_predictions.csv \
        --iou_thresh 0.5 \
        --output_csv results/eval/yolo_detection_metrics.csv
"""
import argparse
import os

import pandas as pd

from src.metrics import box_iou


def evaluate_per_image(gt: pd.DataFrame, preds: pd.DataFrame, iou_thresh: float = 0.5) -> pd.DataFrame:
    gt_group = gt.groupby("filename")
    pred_group = preds.groupby("filename")

    results = []
    for filename, gt_boxes_df in gt_group:
        gt_boxes = gt_boxes_df[["xmin", "ymin", "xmax", "ymax"]].values.tolist()
        pred_boxes = pred_group.get_group(filename)[["xmin", "ymin", "xmax", "ymax"]].values.tolist() if filename in pred_group.groups else []

        matched_gt = set()
        tp = 0
        fp = 0

        for pred_box in pred_boxes:
            best_iou, best_idx = 0.0, -1
            for idx, gt_box in enumerate(gt_boxes):
                if idx in matched_gt:
                    continue
                score = box_iou(pred_box, gt_box)
                if score > best_iou:
                    best_iou, best_idx = score, idx

            if best_iou >= iou_thresh:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1

        fn = len(gt_boxes) - len(matched_gt)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        results.append({"filename": filename, "precision": precision, "recall": recall, "f1_score": f1})

    return pd.DataFrame(results)


def main(gt_csv: str, pred_csv: str, iou_thresh: float, output_csv: str):
    gt = pd.read_csv(gt_csv)
    preds = pd.read_csv(pred_csv)

    # Match by filename stem: gt/pred CSVs don't always share the same image extension.
    gt["filename"] = gt["filename"].apply(lambda x: os.path.splitext(str(x))[0])
    preds["filename"] = preds["filename"].apply(lambda x: os.path.splitext(str(x))[0])

    per_image = evaluate_per_image(gt, preds, iou_thresh)

    print(f"Evaluated {len(per_image)} images at IoU >= {iou_thresh}")
    print(f"Mean Precision: {per_image['precision'].mean():.4f}")
    print(f"Mean Recall:    {per_image['recall'].mean():.4f}")
    print(f"Mean F1-score:  {per_image['f1_score'].mean():.4f}")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        per_image.to_csv(output_csv, index=False)
        print(f"Per-image metrics saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_csv", type=str, required=True, help="Ground-truth boxes (filename, xmin, ymin, xmax, ymax)")
    parser.add_argument("--pred_csv", type=str, required=True, help="Predicted boxes, same format")
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--output_csv", type=str, default=None)
    args = parser.parse_args()

    main(gt_csv=args.gt_csv, pred_csv=args.pred_csv, iou_thresh=args.iou_thresh, output_csv=args.output_csv)