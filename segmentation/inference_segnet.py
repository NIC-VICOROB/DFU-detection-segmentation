import argparse
import os
import numpy as np
import cv2
import tensorflow as tf
from tqdm import tqdm
from segmentation.train_segnet import build_segnet, combo_loss



def restore_size_and_save(predictions, filenames, orig_images_dir,
                          output_dir, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    for pred, fname in zip(predictions, filenames):
        orig = cv2.imread(os.path.join(orig_images_dir, fname), cv2.IMREAD_COLOR)
        if orig is None:
            continue
        orig_h, orig_w = orig.shape[:2]
        mask = (pred[..., 0] > threshold).astype(np.uint8) * 255
        mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_dir, os.path.splitext(fname)[0] + '.png'), mask)



def run_inference(images_dir, weights_path, output_dir,
                  target_size=(480, 640), threshold=0.5):

    model = build_segnet(n_filters=32, num_channels=3)
    model.compile(optimizer='adam', loss=combo_loss)
    model.load_weights(weights_path)

    image_files = sorted([f for f in os.listdir(images_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    predictions, filenames = [], []
    for filename in tqdm(image_files, desc='Running SegNet inference'):
        img = cv2.imread(os.path.join(images_dir, filename), cv2.IMREAD_COLOR)
        img = img.astype('float32') / 255.0
        img_resized = cv2.resize(img, (target_size[1], target_size[0]))
        img_input = np.expand_dims(img_resized, axis=0)
        pred = model.predict(img_input, verbose=0)
        predictions.append(pred[0])
        filenames.append(filename)

    restore_size_and_save(predictions, filenames, images_dir, output_dir, threshold)
    print(f'Masks saved to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, required=True)
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results/segnet_masks')
    parser.add_argument('--target_size', type=int, nargs=2, default=[480, 640])
    parser.add_argument('--threshold', type=float, default=0.5)
    args = parser.parse_args()

    run_inference(
        images_dir=args.images_dir,
        weights_path=args.weights_path,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        threshold=args.threshold
    )