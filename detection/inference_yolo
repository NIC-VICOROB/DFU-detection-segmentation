import argparse
import os
import pandas as pd
import torch
from ultralytics import YOLO
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def run_inference(images_dir, weights_path, output_csv, conf=0.05, imgsz=640):
    model = YOLO(weights_path)
    results = []

    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    for filename in tqdm(image_files, desc='Running YOLO detection inference'):
        img_path = os.path.join(images_dir, filename)
        preds = model(img_path, conf=conf, device=DEVICE, imgsz=imgsz)
        result = preds[0]

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(boxes, scores):
                results.append({
                    'filename': filename,
                    'xmin': box[0], 'ymin': box[1],
                    'xmax': box[2], 'ymax': box[3],
                    'score': score
                })

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='results/yolo_predictions.csv')
    parser.add_argument('--conf', type=float, default=0.05)
    parser.add_argument('--imgsz', type=int, default=640)
    args = parser.parse_args()

    run_inference(
        images_dir=args.images_dir,
        weights_path=args.weights_path,
        output_csv=args.output_csv,
        conf=args.conf,
        imgsz=args.imgsz
    )