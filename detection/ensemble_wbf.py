import argparse
import os
import pandas as pd
import numpy as np
import torch
from torchvision.ops import nms
from ensemble_boxes import weighted_boxes_fusion


# ======================================================
# NMS Ensemble: not used in the paper
# ======================================================
def run_ensemble_nms(csv_paths, iou_thresh=0.5):
    dfs = [pd.read_csv(p) for p in csv_paths]
    all_filenames = sorted(set().union(*[set(df['filename']) for df in dfs]))
    results = []

    for filename in all_filenames:
        boxes, scores = [], []
        for df in dfs:
            preds = df[df['filename'] == filename]
            for _, row in preds.iterrows():
                boxes.append([row['xmin'], row['ymin'], row['xmax'], row['ymax']])
                scores.append(row['score'])

        if not boxes:
            continue

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        scores_tensor = torch.tensor(scores, dtype=torch.float32)
        keep = nms(boxes_tensor, scores_tensor, iou_thresh)

        for idx in keep:
            b = boxes_tensor[idx].tolist()
            results.append({
                'filename': filename,
                'xmin': b[0], 'ymin': b[1], 'xmax': b[2], 'ymax': b[3],
                'score': scores_tensor[idx].item()
            })

    return pd.DataFrame(results)



def run_wbf_ensemble(csv_paths, image_size=640, iou_thr=0.5, skip_box_thr=0.0):
    boxes_list_per_file = []
    scores_list_per_file = []
    all_filenames = set()

    for path in csv_paths:
        df = pd.read_csv(path)
        b_dict, s_dict = {}, {}
        for _, row in df.iterrows():
            fname = row['filename']
            b_dict.setdefault(fname, []).append(
                [row['xmin'], row['ymin'], row['xmax'], row['ymax']]
            )
            s_dict.setdefault(fname, []).append(row['score'])
        boxes_list_per_file.append(b_dict)
        scores_list_per_file.append(s_dict)
        all_filenames.update(b_dict.keys())

    results = []
    for filename in sorted(all_filenames):
        boxes_list, scores_list, labels_list = [], [], []

        for b_dict, s_dict in zip(boxes_list_per_file, scores_list_per_file):
            boxes = np.array(b_dict.get(filename, [])) / image_size
            scores = s_dict.get(filename, [])
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append([0] * len(scores))

        if all(len(b) == 0 for b in boxes_list):
            continue

        boxes, scores, _ = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        boxes = boxes * image_size

        for b, s in zip(boxes, scores):
            results.append({
                'filename': filename,
                'xmin': b[0], 'ymin': b[1], 'xmax': b[2], 'ymax': b[3],
                'score': s
            })

    return pd.DataFrame(results)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_paths', nargs='+', required=True,
                        help='CSVs list with the prediction of each model')
    parser.add_argument('--method', type=str, default='wbf',
                        choices=['wbf', 'nms'])
    parser.add_argument('--image_size', type=int, default=640)
    parser.add_argument('--iou_thr', type=float, default=0.5)
    parser.add_argument('--skip_box_thr', type=float, default=0.0)
    parser.add_argument('--output_file', type=str,
                        default='results/ensemble_predictions.csv')
    args = parser.parse_args()

    if args.method == 'wbf':
        df_result = run_wbf_ensemble(
            csv_paths=args.csv_paths,
            image_size=args.image_size,
            iou_thr=args.iou_thr,
            skip_box_thr=args.skip_box_thr
        )
    else:
        df_result = run_ensemble_nms(
            csv_paths=args.csv_paths,
            iou_thresh=args.iou_thr
        )

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    df_result.to_csv(args.output_file, index=False)
    print(f'Ensemble saved in {args.output_file}')