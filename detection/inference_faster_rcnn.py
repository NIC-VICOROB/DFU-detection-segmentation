import argparse
import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image, ImageDraw
import pandas as pd
from tqdm import tqdm

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_model(weights_path, num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run_inference(images_dir, weights_path, output_csv,
                  conf=0.25, num_classes=2, save_images=False, output_images_dir=None):

    model = load_model(weights_path, num_classes)
    transform = T.Compose([T.ToTensor()])
    results = []

    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])

    for filename in tqdm(image_files, desc='Running Faster R-CNN inference'):
        img_path = os.path.join(images_dir, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img).to(DEVICE)

        with torch.no_grad():
            outputs = model([img_tensor])[0]

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()

        keep = scores >= conf
        boxes, scores = boxes[keep], scores[keep]

        for box, score in zip(boxes, scores):
            results.append({
                'filename': filename,
                'xmin': box[0], 'ymin': box[1],
                'xmax': box[2], 'ymax': box[3],
                'score': score
            })

        if save_images and output_images_dir:
            os.makedirs(output_images_dir, exist_ok=True)
            draw = ImageDraw.Draw(img)
            for box, score in zip(boxes, scores):
                draw.rectangle(list(box), outline='red', width=2)
                draw.text((box[0], box[1] - 10), f'{score:.2f}', fill='red')
            img.save(os.path.join(output_images_dir, filename))

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f'Results saved to {output_csv}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--output_csv', type=str, default='results/faster_rcnn_predictions.csv')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--save_images', action='store_true')
    parser.add_argument('--output_images_dir', type=str, default=None)
    args = parser.parse_args()

    run_inference(
        images_dir=args.images_dir,
        weights_path=args.weights_path,
        output_csv=args.output_csv,
        conf=args.conf,
        num_classes=args.num_classes,
        save_images=args.save_images,
        output_images_dir=args.output_images_dir
    )