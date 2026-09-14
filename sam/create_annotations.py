"""
Build a SAM inference annotations file (JSON) that pairs each mask filename with a
detection bounding box prompt. Used before inference_sam.py / inference_sam_lora.py
to prompt SAM with a specific detector's (or ensemble's) boxes instead of the
ground-truth mask box.

Output format:
{
    "<split_name>": {
        "<filename>.jpg": {"mask_path": "<path to reference mask, for evaluation>", "bbox": [xmin, ymin, xmax, ymax]},
        ...
    }
}

If a filename has no matching row in the detection CSV, its bbox is set to
[0, 0, 0, 0], which downstream inference scripts treat as "no detection" and skip.
"""
import argparse
import json
import os
from glob import glob

import pandas as pd


def load_bbox_dict(csv_path: str) -> dict:
    """Keyed by filename stem (without extension), since masks and detection CSVs
    don't always share the same image extension (e.g. .png masks vs .jpg detections)."""
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


def build_split(mask_dir: str, bbox_dict: dict, mask_ext: str = ".png") -> dict:
    annotations = {}
    mask_paths = sorted(glob(os.path.join(mask_dir, f"*{mask_ext}")))

    for mask_path in mask_paths:
        fname = os.path.basename(mask_path)
        stem = os.path.splitext(fname)[0]
        bbox = bbox_dict.get(stem, [0, 0, 0, 0])
        if stem not in bbox_dict:
            print(f"[WARN] No bbox found for {fname}, using [0, 0, 0, 0].")

        annotations[fname] = {
            "mask_path": mask_path,
            "bbox": bbox,
        }

    return annotations


def main(bbox_csv: str, mask_dir: str, output_json: str, split_name: str, mask_ext: str):
    bbox_dict = load_bbox_dict(bbox_csv)
    data = {split_name: build_split(mask_dir, bbox_dict, mask_ext)}

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Created {output_json} with {len(data[split_name])} images in split '{split_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox_csv", type=str, required=True, help="CSV with detection boxes (filename, xmin, ymin, xmax, ymax)")
    parser.add_argument("--mask_dir", type=str, required=True, help="Folder with reference masks used to list the filenames of this split")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--split_name", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--mask_ext", type=str, default=".png")
    args = parser.parse_args()

    main(
        bbox_csv=args.bbox_csv,
        mask_dir=args.mask_dir,
        output_json=args.output_json,
        split_name=args.split_name,
        mask_ext=args.mask_ext,
    )
