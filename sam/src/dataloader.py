"""
Dataset used to fine-tune SAM with LoRA. Each sample pairs an image with a bounding
box prompt and its ground-truth binary mask.

The bounding box prompt comes from a detection CSV (e.g. YOLO/Faster R-CNN/nnDetection
predictions, or an ensemble of them) so that training mimics the detection-guided
inference setup. If an image is missing from the CSV, the box is instead derived from
the ground-truth mask (see get_bounding_box in utils.py).
"""
import glob
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils import get_bounding_box 

if TYPE_CHECKING:
    # Only needed for type hints; the actual class lives in the external
    # Sam_LoRA dependency (see sam/README.md) and is not required at import time.
    from src.processor import Samprocessor  # noqa: F401


class DatasetSegmentation(Dataset):
    """
    Dataset that pairs images, ground-truth masks and bounding box prompts for SAM.

    Arguments:
        config_file: Parsed lora_config.yaml dictionary
        processor: Samprocessor instance (from the Sam_LoRA dependency) that resizes
            and normalises the image and box prompt to SAM's expected input format
        mode: "train" or "test", selects DATASET.TRAIN_PATH or DATASET.TEST_PATH

    Return:
        dict with keys: image, original_size, boxes, prompt, ground_truth_mask
    """

    def __init__(self, config_file: dict, processor: "Samprocessor", mode: str):
        super().__init__()
        self.processor = processor

        bbox_csv = config_file["DATASET"].get("BBOX_CSV")
        self.bbox_dict = self._load_bbox_csv(bbox_csv) if bbox_csv and os.path.exists(bbox_csv) else {}

        split_path = config_file["DATASET"]["TRAIN_PATH"] if mode == "train" else config_file["DATASET"]["TEST_PATH"]
        self.img_files = sorted(glob.glob(os.path.join(split_path, "images", "*.jpg")))
        self.mask_files = [
            os.path.join(split_path, "masks", os.path.basename(f)) for f in self.img_files
        ]

    @staticmethod
    def _load_bbox_csv(csv_path: str) -> dict:
        """
        Load a detection CSV and keep, per filename, the box with the highest score
        (or the union of all boxes for that filename if no score column is present).
        """
        df = pd.read_csv(csv_path)
        bbox_dict = {}

        if "score" in df.columns:
            for fname, group in df.groupby("filename"):
                best = group.loc[group["score"].idxmax()]
                bbox_dict[fname] = [best["xmin"], best["ymin"], best["xmax"], best["ymax"]]
        else:
            for fname, group in df.groupby("filename"):
                bbox_dict[fname] = [
                    int(group["xmin"].min()),
                    int(group["ymin"].min()),
                    int(group["xmax"].max()),
                    int(group["ymax"].max()),
                ]

        return bbox_dict

    def get_box_prompt(self, filename: str, ground_truth_mask: np.ndarray) -> list:
        """Return the detection box for this image, falling back to the mask bounding box."""
        if filename in self.bbox_dict:
            return self.bbox_dict[filename]
        return get_bounding_box(ground_truth_mask)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index: int) -> dict:
        img_path = self.img_files[index]
        mask_path = self.mask_files[index]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        ground_truth_mask = (np.array(mask) > 0).astype(np.uint8)
        original_size = tuple(image.size)[::-1]  # (H, W)

        filename = os.path.basename(img_path)
        box = self.get_box_prompt(filename, ground_truth_mask)

        inputs = self.processor(image, original_size, box)
        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask)

        return inputs


def collate_fn(batch) -> list:
    """Keep the batch as a list of dicts (SAM's LoRA processor expects this, not a stacked tensor)."""
    return list(batch)
