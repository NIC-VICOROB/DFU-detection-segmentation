import argparse
import os
import cv2
import json
from tqdm import tqdm


def create_coco_annotations(image_dir, mask_dir, output_file):
    images, annotations, categories = [], [], [{
        'id': 1,
        'name': 'ulcer',
        'supercategory': 'none'
    }]
    annotation_id, image_id = 1, 1

    for filename in tqdm(sorted(os.listdir(image_dir)), desc='Creating COCO annotations'):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + '.png')
        if not os.path.exists(mask_path):
            print(f'Warning: mask not found for {filename}, skipping.')
            continue

        img = cv2.imread(os.path.join(image_dir, filename))
        if img is None:
            continue
        height, width = img.shape[:2]

        images.append({'id': image_id, 'file_name': filename,
                       'width': width, 'height': height})

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            if len(cnt) < 3:
                continue
            segmentation = cnt.flatten().tolist()
            if len(segmentation) < 6:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': 1,
                'segmentation': [segmentation],
                'area': float(cv2.contourArea(cnt)),
                'bbox': [x, y, w, h],
                'iscrowd': 0
            })
            annotation_id += 1

        image_id += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({'images': images, 'annotations': annotations,
                   'categories': categories}, f, indent=4)
    print(f'COCO annotations saved to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--mask_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    create_coco_annotations(args.image_dir, args.mask_dir, args.output_file)