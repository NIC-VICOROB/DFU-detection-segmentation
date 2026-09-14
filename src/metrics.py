import numpy as np


def dice_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dice score between two binary masks."""
    intersection = np.sum(y_true * y_pred)
    return (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)


def mask_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """IoU between two binary masks."""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / (union + 1e-8)


def box_iou(box_a: list, box_b: list) -> float:
    """IoU between two boxes in [xmin, ymin, xmax, ymax] format."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)
    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return inter_area / float(area_a + area_b - inter_area)