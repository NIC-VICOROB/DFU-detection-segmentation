import argparse
import os
import shutil
import numpy as np
import imageio.v2 as imageio
from glob import glob
from tqdm import tqdm


def create_nnunet_dataset(images_dir, output_images_dir,
                          masks_dir=None, output_labels_dir=None,
                          img_ext='.jpg'):

    os.makedirs(output_images_dir, exist_ok=True)
    if output_labels_dir:
        os.makedirs(output_labels_dir, exist_ok=True)

    image_paths = sorted(glob(os.path.join(images_dir, f'*{img_ext}')))
    print(f'Found {len(image_paths)} images.')

    for img_path in tqdm(image_paths, desc='Converting dataset'):
        filename = os.path.basename(img_path)
        base_id = os.path.splitext(filename)[0]

        if masks_dir is not None:
            mask_path = os.path.join(masks_dir, f'{base_id}.png')
            if not os.path.exists(mask_path):
                print(f'Warning: mask not found for {filename}, skipping.')
                continue
            mask = imageio.imread(mask_path)
            mask = (mask > 0).astype(np.uint8)
            imageio.imwrite(os.path.join(output_labels_dir, f'{base_id}.png'), mask)

        shutil.copy(img_path, os.path.join(output_images_dir, f'{base_id}_0000.png'))

    print(f'Dataset conversion complete. {len(image_paths)} cases saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--output_images_dir', type=str, required=True)
    parser.add_argument('--masks_dir', type=str, default=None)
    parser.add_argument('--output_labels_dir', type=str, default=None)
    parser.add_argument('--img_ext', type=str, default='.jpg',
                        choices=['.jpg', '.png', '.jpeg'])
    args = parser.parse_args()

    create_nnunet_dataset(
        images_dir=args.images_dir,
        output_images_dir=args.output_images_dir,
        masks_dir=args.masks_dir,
        output_labels_dir=args.output_labels_dir,
        img_ext=args.img_ext
    )