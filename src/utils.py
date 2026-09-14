import subprocess

import numpy as np


def get_bounding_box(ground_truth_map: np.array) -> list:
    """
    Get the bounding box of a binary mask, with a small random perturbation.

    Arguments:
        ground_truth_map: Binary mask in array format

    Return:
        bbox: Bounding box of the mask [X, Y, X, Y]
    """
    idx = np.where(ground_truth_map > 0)

    if idx[0].size == 0 or idx[1].size == 0:
        return [0, 0, 0, 0]

    x_indices = idx[1]
    y_indices = idx[0]

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

    H, W = ground_truth_map.shape
    x_min = max(0, x_min - np.random.randint(0, 20))
    x_max = min(W, x_max + np.random.randint(0, 20))
    y_min = max(0, y_min - np.random.randint(0, 20))
    y_max = min(H, y_max + np.random.randint(0, 20))

    return [x_min, y_min, x_max, y_max]


def get_gpu_usage():
    """
    Returns:
        List[Tuple[int, int, int, int]]: (gpu_index, memory_used, memory_free, memory_total) in MB
    """
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,memory.total", "--format=csv,noheader,nounits"]
        )
        gpu_usage = [line.split(',') for line in output.decode('utf-8').strip().split('\n')]
        return [(int(index), int(used), int(free), int(total)) for index, used, free, total in gpu_usage]
    except subprocess.CalledProcessError as e:
        print("Error running nvidia-smi:", e)
        return []


def get_least_used_gpu():
    """
    Returns:
        int: Index of the GPU with the least memory used, or -1 if none found.
    """
    gpu_usage = get_gpu_usage()
    least_used_index = -1
    least_memory_used = float('inf')

    for index, used, free, total in gpu_usage:
        print(f"GPU {index}: used {used} MB, free {free} MB, total {total} MB")
        if used < least_memory_used:
            least_memory_used = used
            least_used_index = index

    return least_used_index