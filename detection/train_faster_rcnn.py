import argparse
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import torchvision.transforms as T

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CocoDataset(Dataset):
    def __init__(self, images_dir, annotation_file, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        self.coco = COCO(annotation_file)
        self.ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])

        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([img_id])
        }

        if self.transforms:
            img = self.transforms(img)

        return img, target



def evaluate(model, data_loader, device, coco_gt):
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = int(target["image_id"])
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()
                labels = output["labels"].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    results.append({
                        "image_id": image_id,
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })

    coco_dt = coco_gt.loadRes(results) if results else coco_gt.loadRes([])
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = [0.5]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0]



def train(train_images, train_ann, val_images, val_ann,
          num_classes=2, epochs=200, lr=0.005, batch_size=4,
          output_dir='weights/faster_rcnn'):

    os.makedirs(output_dir, exist_ok=True)
    transform = T.Compose([T.ToTensor()])

    train_dataset = CocoDataset(train_images, train_ann, transform)
    val_dataset = CocoDataset(val_images, val_ann, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda x: tuple(zip(*x)))

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(DEVICE)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    coco_val_gt = COCO(val_ann)
    best_map = 0.0
    metrics_log = {}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        lr_scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        map50 = evaluate(model, val_loader, DEVICE, coco_val_gt)
        metrics_log[f"epoch_{epoch+1}"] = {"loss": avg_loss, "mAP@0.5": map50}

        torch.save(model.state_dict(), os.path.join(output_dir, "last_model.pth"))
        if map50 > best_map:
            best_map = map50
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | mAP@0.5: {map50:.4f}")

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics_log, f, indent=4)

    print(f"Training finished. Best mAP@0.5: {best_map:.4f}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images', type=str, required=True)
    parser.add_argument('--train_ann', type=str, required=True)
    parser.add_argument('--val_images', type=str, required=True)
    parser.add_argument('--val_ann', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='weights/faster_rcnn')
    args = parser.parse_args()

    train(
        train_images=args.train_images,
        train_ann=args.train_ann,
        val_images=args.val_images,
        val_ann=args.val_ann,
        num_classes=args.num_classes,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        output_dir=args.output_dir
    )