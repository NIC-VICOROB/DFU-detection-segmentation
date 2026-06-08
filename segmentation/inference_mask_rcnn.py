import argparse
import os
import numpy as np
import cv2
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
from mmengine.config import Config
from tqdm import tqdm


def run_inference(images_dir, config_path, weights_path, output_dir, conf=0.25):
    os.makedirs(output_dir, exist_ok=True)

    cfg = Config.fromfile(config_path)
    model = init_detector(cfg, weights_path, device='cuda:0')

    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for filename in tqdm(image_files, desc='Running Mask R-CNN inference'):
        img_path = os.path.join(images_dir, filename)
        img = cv2.imread(img_path)
        result = inference_detector(model, img)

        mask_name = os.path.splitext(filename)[0] + '.png'
        pred_masks = result.pred_instances.masks
        scores = result.pred_instances.scores.cpu().numpy()

        if pred_masks is not None and len(pred_masks) > 0:
            keep = scores >= conf
            masks = pred_masks[keep].data.cpu().numpy()
            if len(masks) > 0:
                combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
            else:
                combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        else:
            combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        cv2.imwrite(os.path.join(output_dir, mask_name), combined_mask)

    print(f'Masks saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--config_path', type=str, default='configs/mask_rcnn_ulcer.py')
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/mask_rcnn_masks')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()

    run_inference(
        images_dir=args.images_dir,
        config_path=args.config_path,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        conf=args.conf
    )