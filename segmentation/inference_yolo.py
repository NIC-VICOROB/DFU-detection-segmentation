import argparse
import os
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_inference(images_dir, weights_path, output_dir, conf=0.05, imgsz=640):
    model = YOLO(weights_path)
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for filename in tqdm(image_files, desc='Running YOLO segmentation inference'):
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        preds = model(img_path, conf=conf, device=DEVICE, imgsz=imgsz)
        result = preds[0]

        mask_name = os.path.splitext(filename)[0] + '.png'

        if result.masks is not None:
            mask = result.masks[0].data.cpu().squeeze(0).numpy()
        else:
            mask = np.zeros((img.shape[0], img.shape[1]))

        mask = (mask > 0).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(output_dir, mask_name), mask)

    print(f'Masks saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/yolo_seg_masks')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()

    run_inference(
        images_dir=args.images_dir,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        conf=args.conf,
        imgsz=args.imgsz
    )