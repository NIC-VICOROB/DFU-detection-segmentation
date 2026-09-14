import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import torch


def show_mask(mask: np.array, ax, random_color=False):
    """Plot the mask"""
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[:2]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def plot_image_mask(image: PIL.Image, mask: PIL.Image, filename: str):
    """Plot the image and the mask superposed"""
    fig, axes = plt.subplots()
    axes.imshow(np.array(image))
    ground_truth_seg = np.array(mask)
    show_mask(ground_truth_seg, axes)
    axes.title.set_text(f"{filename} predicted mask")
    axes.axis("off")
    plt.savefig("./plots/" + filename + ".jpg")
    plt.close()


def plot_image_mask_dataset(dataset: torch.utils.data.Dataset, idx: int):
    """Take an image from the dataset and plot it"""
    image_path = dataset.img_files[idx]
    mask_path = dataset.mask_files[idx]
    image = Image.open(image_path)
    mask = Image.open(mask_path)
    mask = mask.convert('1')
    plot_image_mask(image, mask)


def stacking_batch(batch, outputs):
    """
    Stacks the batch ground-truth masks and SAM's low_res_logits outputs to compute the loss.

    Return:
        stk_gt: [batch_size, H, W]
        stk_out: [batch_size, 1, 1, H, W]
    """
    stk_gt = torch.stack([b["ground_truth_mask"] for b in batch], dim=0)
    stk_out = torch.stack([out["low_res_logits"] for out in outputs], dim=0)

    return stk_gt, stk_out