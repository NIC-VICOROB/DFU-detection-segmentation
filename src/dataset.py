"""
Shared Keras-compatible paired image/mask data generator, used by the Keras-based
segmentation models (SegNet, FUSegNet). Detection and other segmentation models
(YOLO, Mask R-CNN, nnUNet) use their own framework's native dataset format instead.
"""
import os

import cv2
import numpy as np
from tensorflow.keras.utils import Sequence


class PairedImageMaskGenerator(Sequence):
    """
    Loads image/mask pairs from disk, normalising images to [0, 1] and masks to
    binary {0, 1}, resized to target_size.

    Arguments:
        images_path: Folder with .jpg images
        labels_path: Folder with .png masks (same filenames, .png extension)
        batch_size: Batch size
        target_size: (H, W) to resize both image and mask to
    """

    def __init__(self, images_path: str, labels_path: str, batch_size: int, target_size: tuple):
        self.images_path = images_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.target_size = target_size

        self.image_files = sorted(f for f in os.listdir(images_path) if f.endswith(".jpg"))
        self.label_files = sorted(f for f in os.listdir(labels_path) if f.endswith(".png"))

        if len(self.image_files) != len(self.label_files):
            raise ValueError(
                f"Mismatch: {len(self.image_files)} images and {len(self.label_files)} masks."
            )

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, idx):
        batch_images = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.label_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []

        for img_file, mask_file in zip(batch_images, batch_masks):
            img = cv2.imread(os.path.join(self.images_path, img_file), cv2.IMREAD_COLOR)
            mask = cv2.imread(os.path.join(self.labels_path, mask_file), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue

            img = img.astype("float32") / 255.0
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))

            mask = (mask > 0).astype("float32")
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)