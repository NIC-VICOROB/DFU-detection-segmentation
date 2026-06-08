from mmengine.config import Config
from mmengine.runner import set_random_seed

# Load base config
cfg = Config.fromfile('mmdetection/configs/mask_rcnn/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco.py')

# Dataset
cfg.metainfo = {
    'classes': ('ulcer',),
    'palette': [(220, 20, 60)]
}

data_root = 'data/DFUC/'

cfg.train_dataloader.dataset.ann_file = 'annotations/instances_train.json'
cfg.train_dataloader.dataset.data_root = data_root
cfg.train_dataloader.dataset.data_prefix.img = 'images/train'
cfg.train_dataloader.dataset.metainfo = cfg.metainfo

cfg.val_dataloader.dataset.ann_file = 'annotations/instances_val.json'
cfg.val_dataloader.dataset.data_root = data_root
cfg.val_dataloader.dataset.data_prefix.img = 'images/val'
cfg.val_dataloader.dataset.metainfo = cfg.metainfo

cfg.test_dataloader = cfg.val_dataloader
cfg.val_evaluator.ann_file = f'{data_root}/annotations/instances_val.json'
cfg.test_evaluator = cfg.val_evaluator

# Model
cfg.model.roi_head.bbox_head.num_classes = 1
cfg.model.roi_head.mask_head.num_classes = 1
cfg.load_from = 'weights/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco.pth'

# Training
cfg.work_dir = 'weights/mask_rcnn'
cfg.train_cfg.val_interval = 3
cfg.default_hooks.checkpoint.interval = 3
cfg.optim_wrapper.optimizer.lr = 1e-3
cfg.default_hooks.logger.interval = 10

set_random_seed(0, deterministic=False)