# Data

## Datasets

| Dataset | Access | Images | Notes |
|---|---|---|---|
| **DFUC2020** | Request via [dfu-challenge.github.io](https://dfu-challenge.github.io) | 2,000 train / 200 val / 2,000 test | Non-commercial research use only |
| **DFUC2022** | Request via [dfu-challenge.github.io](https://dfu-challenge.github.io) | 2,000 train / 2,000 test | |
| **DFUC2024** | Request via [dfu-challenge.github.io](https://dfu-challenge.github.io) | See challenge page | 4th edition, self-supervised learning focus |
| **FUSeg** | Freely available, no registration: [uwm-bigdata/wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation/tree/master/data/Foot%20Ulcer%20Segmentation%20Challenge) | 810 train / 200 val / 200 test | Used only for **out-of-domain evaluation** (no fine-tuning), see paper |

The DFU Challenge portal (`dfu-challenge.github.io`) handles access requests for all
DFUC editions; approval and turnaround time depend on the organisers.

## Expected folder structure

After downloading, arrange each dataset as:
```
data/
├── DFUC2020/
│ ├── images/{train,val,test}/.jpg
│ └── masks/{train,val,test}/.png # segmentation ground truth
├── DFUC2022/
│ └── ... (same structure)
├── DFUC2024/
│ └── ... (same structure)
└── FUSeg/
├── images/{train,val,test}/.jpg
└── masks/{train,val,test}/.png
```


`splits.json` (repository root) defines the 5-fold train/val split used in the paper,
by filename — use it instead of re-splitting with a random seed, to reproduce the
paper's results exactly.

## Framework-specific formats

Most scripts read the raw `images/` + `masks/` layout above directly. A few
frameworks need a converted copy instead — see the corresponding script rather than
converting by hand:

- **nnU-Net** (segmentation): `segmentation/convert_to_nnunet.py`
- **nnDetection**: `detection/convert_to_nndetection.py` (2D images -> pseudo-3D NIfTI)
- **Mask R-CNN** (COCO-format annotations): `segmentation/create_coco_annotations.py`
- **YOLO detection/segmentation**: point a `configs/*.yaml` at the raw folders (Ultralytics reads YOLO-format labels directly); use `src/preprocessing.py masks_to_yolo` to generate YOLO-segmentation `.txt` labels from the binary masks

## Detection labels

Detection scripts expect a `labels_detection/` folder alongside `images/`, with one
`.txt` file per image in YOLO detection format (`class cx cy w h`, normalised).
