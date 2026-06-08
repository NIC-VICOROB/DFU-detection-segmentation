import argparse
import os
import json
import numpy as np
import nibabel as nib
from PIL import Image
from tqdm import tqdm


def replicate_image_volume(img, z_depth=96):
    # img shape: (H, W, 3) or (H, W)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = np.mean(img, axis=2)  # grayscale
    volume = np.stack([img] * z_depth, axis=0)  # (Z, H, W)
    return volume


def convert_dataset(images_dir, labels_dir, output_dir, z_depth=96):
    images_out = os.path.join(output_dir, 'raw_splitted', 'imagesTr')
    labels_out = os.path.join(output_dir, 'raw_splitted', 'labelsTr')
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.endswith(('.jpg', '.jpeg', '.png'))])

    for i, fname in enumerate(tqdm(image_files, desc='Converting images')):
        case_id = f'case{i:04d}'

        # Convert image
        img = np.array(Image.open(os.path.join(images_dir, fname)))
        volume = replicate_image_volume(img, z_depth)
        nib.save(
            nib.Nifti1Image(volume.astype(np.float32), np.eye(4)),
            os.path.join(images_out, f'{case_id}_0000.nii.gz')
        )

        # Convert label (bounding box -> instance segmentation mask)
        label_file = os.path.join(labels_dir, fname.replace('.jpg', '.txt'))
        instance_mask = np.zeros(volume.shape, dtype=np.uint8)
        instances = {}

        if os.path.exists(label_file):
            h, w = img.shape[:2]
            with open(label_file) as f:
                for inst_idx, line in enumerate(f, start=1):
                    _, cx, cy, bw, bh = map(float, line.strip().split())
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    instance_mask[:, y1:y2, x1:x2] = inst_idx
                    instances[str(inst_idx)] = 0  # class 0: ulcer

        nib.save(
            nib.Nifti1Image(instance_mask, np.eye(4)),
            os.path.join(labels_out, f'{case_id}.nii.gz')
        )
        with open(os.path.join(labels_out, f'{case_id}.json'), 'w') as f:
            json.dump({'instances': instances}, f)

    # dataset.json
    dataset_info = {
        'task': 'Task100_DFU',
        'name': 'Diabetic Foot Ulcer Detection',
        'dim': 3,
        'target_class': 0,
        'test_labels': False,
        'labels': {'0': 'ulcer'},
        'modalities': {'0': 'RGB'}
    }
    with open(os.path.join(output_dir, 'dataset.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)

    print(f'Conversion complete. {len(image_files)} cases saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--z_depth', type=int, default=96)
    args = parser.parse_args()

    convert_dataset(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        z_depth=args.z_depth
    )