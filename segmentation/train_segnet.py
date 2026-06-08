import argparse
import os
import json
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K



class DFUGenerator:
    def __init__(self, images_path, labels_path, batch_size, target_size):
        self.images_path = images_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.target_size = target_size  # (H, W)

        self.image_files = sorted([f for f in os.listdir(images_path)
                                   if f.endswith('.jpg')])
        self.label_files = sorted([f for f in os.listdir(labels_path)
                                   if f.endswith('.png')])

        if len(self.image_files) != len(self.label_files):
            raise ValueError(f'Mismatch: {len(self.image_files)} images '
                             f'and {len(self.label_files)} masks.')

    def __len__(self):
        return len(self.image_files) // self.batch_size

    def __getitem__(self, idx):
        batch_imgs = self.image_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_masks = self.label_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        X, Y = [], []

        for img_file, mask_file in zip(batch_imgs, batch_masks):
            img = cv2.imread(os.path.join(self.images_path, img_file), cv2.IMREAD_COLOR)
            mask = cv2.imread(os.path.join(self.labels_path, mask_file), cv2.IMREAD_GRAYSCALE)

            if img is None or mask is None:
                continue

            img = img.astype('float32') / 255.0
            mask = (mask > 0).astype('float32')
            mask = np.expand_dims(mask, axis=-1)

            img = cv2.resize(img, (self.target_size[1], self.target_size[0]))
            mask = cv2.resize(mask, (self.target_size[1], self.target_size[0]),
                              interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)

            X.append(img)
            Y.append(mask)

        return np.array(X), np.array(Y)



def dice_loss(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def combo_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)



def build_segnet(n_filters=32, num_channels=3):
    inp = Input(shape=(None, None, num_channels))

    # Encoder
    x = Conv2D(n_filters, 9, activation='relu', padding='same')(inp)
    x = MaxPooling2D()(x)
    x = Conv2D(n_filters, 5, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(n_filters * 2, 5, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(n_filters * 2, 5, activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(n_filters * 2, 5, activation='relu', padding='same')(x)

    # Decoder
    x = UpSampling2D()(x)
    x = Conv2D(n_filters, 7, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(n_filters, 5, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    x = Conv2D(n_filters, 5, activation='relu', padding='same')(x)
    x = UpSampling2D()(x)
    out = Conv2D(1, 1, activation='sigmoid', padding='same')(x)

    return Model(inp, out)



def train(train_images, train_labels, val_images, val_labels,
          output_dir='weights/segnet', target_size=(480, 640),
          batch_size=4, epochs=50, n_filters=32):

    os.makedirs(output_dir, exist_ok=True)

    model = build_segnet(n_filters=n_filters, num_channels=3)
    model.compile(optimizer='adam', loss=combo_loss, metrics=['accuracy'])

    train_gen = DFUGenerator(train_images, train_labels, batch_size, target_size)
    val_gen = DFUGenerator(val_images, val_labels, batch_size, target_size)

    checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'best_model.h5'),
        save_best_only=True,
        monitor='val_loss'
    )

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=[checkpoint]
    )

    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history.history, f, indent=2)

    print(f'Training complete. Best model saved to {output_dir}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images', type=str, required=True)
    parser.add_argument('--train_labels', type=str, required=True)
    parser.add_argument('--val_images', type=str, required=True)
    parser.add_argument('--val_labels', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='weights/segnet')
    parser.add_argument('--target_size', type=int, nargs=2, default=[480, 640])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--n_filters', type=int, default=32)
    args = parser.parse_args()

    train(
        train_images=args.train_images,
        train_labels=args.train_labels,
        val_images=args.val_images,
        val_labels=args.val_labels,
        output_dir=args.output_dir,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size,
        epochs=args.epochs,
        n_filters=args.n_filters
    )