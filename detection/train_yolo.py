import argparse
import os
import torch
from ultralytics import YOLO

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train(config='yolo_baseline', epochs=200, lr0=0.001, patience=20, imgsz=640, output_dir='weights/yolo_det'):
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO('yolo12n.yaml')
    model.to(device=DEVICE)
    
    config_path = os.path.join('configs', f'{config}.yaml') # 'clean_code_paper1', 'DFU-detection-segmentation', 
    
    results = model.train(
        data=config_path,
        epochs=epochs,
        device=DEVICE,
        lr0=lr0,
        patience=patience,
        imgsz=imgsz,
        project=output_dir,
        name=config
    )
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='yolo_baseline')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--output_dir', type=str, default='weights/yolo_det')
    args = parser.parse_args()
    
    train(
        config=args.config,
        epochs=args.epochs,
        lr0=args.lr0,
        patience=args.patience,
        imgsz=args.imgsz,
        output_dir=args.output_dir
    )