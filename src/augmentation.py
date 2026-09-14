"""
Strong augmentation pipeline for training data, with extra augmentations for images
containing small ulcers (harder to learn, so oversampled).

Requires albumentations: pip install albumentations
"""
import argparse
import os

import cv2
import numpy as np
from tqdm import tqdm

try:
    import albumentations as A
except ImportError as e:
    raise ImportError("This module requires albumentations. Install with: pip install albumentations") from e


SMALL_ULCER_AREA_THRESHOLD = 300  # px^2; ulcers below this are considered "small"

TRANSFORM = A.Compose([
    A.Rotate(limit=90, p=0.8),
    A.RandomScale(scale_limit=(0.6, 1.4), p=0.8),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.7),
    A.HorizontalFlip(p=0.7),
    A.VerticalFlip(p=0.4),
    A.ElasticTransform(alpha=2, sigma=50, alpha_affine=30, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.5, p=0.5),
    A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.6),
    A.MedianBlur(blur_limit=5, p=0.6),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.ImageCompression(quality_lower=40, quality_upper=80, p=0.5),
    A.Downscale(scale_min=0.5, scale_max=0.9, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.6),
    A.CoarseDropout(max_holes=5, max_height=20, max_width=20, min_holes=1, p=0.4),
], additional_targets={"mask": "mask"})


def has_small_ulcer(mask: np.ndarray, threshold: int = SMALL_ULCER_AREA_THRESHOLD) -> bool:
    binary = (mask > 0).astype(np.uint8)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return any(stats[i, cv2.CC_STAT_AREA] <= threshold for i in range(1, num_labels))


def save_pair(image: np.ndarray, mask: np.ndarray, images_dir: str, masks_dir: str, name: str, index):
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    cv2.imwrite(os.path.join(images_dir, f"{name}_{index}.jpg"), image)
    cv2.imwrite(os.path.join(masks_dir, f"{name}_{index}.png"), mask)


def augment_dataset(images_dir: str, masks_dir: str, output_images_dir: str, output_masks_dir: str,
                     num_aug: int = 14, num_small_ulcer_extra: int = 15,
                     small_ulcer_threshold: int = SMALL_ULCER_AREA_THRESHOLD):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]

    for image_file in tqdm(image_files, desc="Augmenting dataset"):
        name = os.path.splitext(image_file)[0]

        image = cv2.imread(os.path.join(images_dir, image_file))
        mask = cv2.imread(os.path.join(masks_dir, image_file.replace(".jpg", ".png")), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            print(f"[WARN] Could not read image/mask pair for {image_file}, skipping.")
            continue

        # Keep the original, unaugmented pair
        save_pair(image, mask, output_images_dir, output_masks_dir, name, "orig")

        n_aug = num_aug + (num_small_ulcer_extra if has_small_ulcer(mask, small_ulcer_threshold) else 0)

        for i in range(1, n_aug + 1):
            augmented = TRANSFORM(image=image, mask=mask)
            save_pair(augmented["image"], augmented["mask"], output_images_dir, output_masks_dir, name, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--masks_dir", type=str, required=True)
    parser.add_argument("--output_images_dir", type=str, required=True)
    parser.add_argument("--output_masks_dir", type=str, required=True)
    parser.add_argument("--num_aug", type=int, default=14)
    parser.add_argument("--num_small_ulcer_extra", type=int, default=15)
    parser.add_argument("--small_ulcer_threshold", type=int, default=SMALL_ULCER_AREA_THRESHOLD)
    args = parser.parse_args()

    augment_dataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        output_images_dir=args.output_images_dir,
        output_masks_dir=args.output_masks_dir,
        num_aug=args.num_aug,
        num_small_ulcer_extra=args.num_small_ulcer_extra,
        small_ulcer_threshold=args.small_ulcer_threshold,
    )