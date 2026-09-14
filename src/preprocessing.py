"""
Image preprocessing utilities shared across detection/segmentation pipelines:
contrast enhancement (CLAHE + histogram equalisation) and converting binary masks
into YOLO-segmentation polygon .txt annotations.
"""
import argparse
import os

import cv2
from tqdm import tqdm


def enhance_contrast(img_path: str):
    """CLAHE followed by histogram equalisation on the grayscale image."""
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Could not read image: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.equalizeHist(enhanced)


def preprocess_dataset(images_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for filename in tqdm(image_files, desc="Enhancing contrast"):
        enhanced = enhance_contrast(os.path.join(images_dir, filename))
        if enhanced is not None:
            cv2.imwrite(os.path.join(output_dir, filename), enhanced)


def mask_to_yolo_polygon(mask_path: str, output_txt_path: str, class_index: int = 0):
    """Convert a binary mask into a YOLO-segmentation .txt (normalised polygon per contour)."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"[WARN] Could not read mask: {mask_path}")
        return

    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    lines = []
    for contour in contours:
        normalized = [f"{pt[0][0] / width:.6f} {pt[0][1] / height:.6f}" for pt in contour]
        lines.append(f"{class_index} " + " ".join(normalized))

    with open(output_txt_path, "w") as f:
        f.write("\n".join(lines))


def masks_to_yolo_polygons(masks_dir: str, output_dir: str, class_index: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    mask_files = [f for f in os.listdir(masks_dir) if f.endswith(".png")]

    for filename in tqdm(mask_files, desc="Converting masks to YOLO polygons"):
        output_txt_path = os.path.join(output_dir, filename.replace(".png", ".txt"))
        mask_to_yolo_polygon(os.path.join(masks_dir, filename), output_txt_path, class_index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    contrast_parser = subparsers.add_parser("enhance_contrast", help="CLAHE + histogram equalisation on a folder of images")
    contrast_parser.add_argument("--images_dir", type=str, required=True)
    contrast_parser.add_argument("--output_dir", type=str, required=True)

    yolo_parser = subparsers.add_parser("masks_to_yolo", help="Convert a folder of binary masks to YOLO-segmentation .txt files")
    yolo_parser.add_argument("--masks_dir", type=str, required=True)
    yolo_parser.add_argument("--output_dir", type=str, required=True)
    yolo_parser.add_argument("--class_index", type=int, default=0)

    args = parser.parse_args()

    if args.command == "enhance_contrast":
        preprocess_dataset(args.images_dir, args.output_dir)
    elif args.command == "masks_to_yolo":
        masks_to_yolo_polygons(args.masks_dir, args.output_dir, args.class_index)