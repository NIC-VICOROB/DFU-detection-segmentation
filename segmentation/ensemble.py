import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm


def load_mask(path):
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return (m > 0).astype(np.uint8) if m is not None else None


def remove_small_components(mask, min_area=50):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 1
    return clean


def ensemble_priority(m1, m2):
    """
    Priority-based ensemble: m1 is the primary model.
    Components from m2 are only added if they do not overlap with m1.
    """
    ens = m1.copy()
    num2, lab2, _, _ = cv2.connectedComponentsWithStats(m2, connectivity=8)
    for i in range(1, num2):
        comp = (lab2 == i).astype(np.uint8)
        if not np.logical_and(comp, m1).any():
            ens = np.logical_or(ens, comp).astype(np.uint8)
    return ens



def run_ensemble(mask1_dir, mask2_dir, output_dir,
                 method='or', min_area=100, adaptive_min_area=False):

    os.makedirs(output_dir, exist_ok=True)
    filenames = sorted([f for f in os.listdir(mask1_dir) if f.endswith('.png')])

    for fname in tqdm(filenames, desc=f'Ensemble [{method}]'):
        p1 = os.path.join(mask1_dir, fname)
        p2 = os.path.join(mask2_dir, fname)

        m1 = load_mask(p1) if os.path.exists(p1) else None
        m2 = load_mask(p2) if os.path.exists(p2) else None

        # Resize m2 to m1 size if needed
        if m1 is not None and m2 is not None:
            if m1.shape != m2.shape:
                m2 = cv2.resize(m2, (m1.shape[1], m1.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

        # Apply ensemble
        if m1 is not None and m2 is not None:
            if method == 'or':
                ens = np.logical_or(m1, m2).astype(np.uint8)
            elif method == 'and':
                ens = np.logical_and(m1, m2).astype(np.uint8)
            elif method == 'priority':
                ens = ensemble_priority(m1, m2)
        elif m1 is not None:
            ens = m1.copy()
        elif m2 is not None:
            ens = m2.copy()
        else:
            continue

        # Postprocessing: remove small components
        area = int(0.001 * ens.shape[0] * ens.shape[1]) if adaptive_min_area else min_area
        ens = remove_small_components(ens, min_area=area)

        cv2.imwrite(os.path.join(output_dir, fname), ens * 255)

    print(f'Ensemble masks saved to {output_dir}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask1_dir', type=str, required=True,
                        help='Directory with masks from model 1 (primary for priority)')
    parser.add_argument('--mask2_dir', type=str, required=True,
                        help='Directory with masks from model 2 (secondary for priority)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--method', type=str, default='or',
                        choices=['or', 'and', 'priority'])
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--adaptive_min_area', action='store_true',
                        help='Use adaptive min area (0.1% of image size)')
    args = parser.parse_args()

    run_ensemble(
        mask1_dir=args.mask1_dir,
        mask2_dir=args.mask2_dir,
        output_dir=args.output_dir,
        method=args.method,
        min_area=args.min_area,
        adaptive_min_area=args.adaptive_min_area
    )