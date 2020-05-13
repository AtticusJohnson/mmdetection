# model

_base_ = ['../configs/cascade_rcnn/cascade_rcnn_r50_fpn_20e_coco.py',
          # '../configs/_base_/datasets/coco_detection.py',
          # '../configs/_base_/schedules/schedule_1x.py',
          ]
model = dict(
    type='CascadeRCNN',
    pretrained='open-mmlab://resnext101_64x4d',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        style='pytorch',
        dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True))
)
# model = dict(
#     backbone=dict(
#         dcn=dict(type='DCN', deformable_groups=1, fallback_on_stride=False),
#         stage_with_dcn=(False, True, True, True)))

# model = dict(
#     type='CascadeRCNN',
#     pretrained='open-mmlab://resnext101_64x4d',
#     backbone=dict(
#         type='ResNeXt',
#         depth=101,
#         groups=64,
#         base_width=4,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         style='pytorch',
#         )
# )

# dataset
classes = ('wheat', )  # only 1 class
dataset_type = 'CocoDataset'
use_spec_eval_index = "wheat_eval"  # add specific eval index if necessary.  # add specific eval index if necessary.
data_root = '../wheat/'
img_norm_cfg = dict(mean=[80.232, 80.940, 54.676], std=[53.058, 53.754, 45.068], to_rgb=True)

# OneOf = [dict(type='RandomContrast'),
#          dict(type='RandomGamma'),
#          dict(type='RandomBrightness')]
# trans = [
#     dict(type='RandomSizedBBoxSafeCrop', height=1024, width=1024, erosion_rate=0.0, interpolation=1, p=1.0),
#     dict(type='HorizontalFlip', p=0.5),
#     dict(type='VerticalFlip', p=0.5),
#     dict(type='OneOf', transforms=[dict(type='RandomContrast'),
#                                    dict(type='RandomGamma'),
#                                    dict(type='RandomBrightness')], p=1.0),
#     dict(type='CLAHE', p=1.0)]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Albu', transforms=[
        dict(type='RandomSizedBBoxSafeCrop', height=1024, width=1024, erosion_rate=0.0, interpolation=1, p=1.0),
        dict(type='HorizontalFlip', p=0.5),
        dict(type='VerticalFlip', p=0.5),
        dict(type='OneOf', transforms=[dict(type='RandomContrast'),
                                       dict(type='RandomGamma'),
                                       dict(type='RandomBrightness')], p=1.0),
        dict(type='CLAHE', p=1.0)],
         bbox_params=dict(type='BboxParams', format='pascal_voc', label_fields=['gt_labels'])),
    dict(type='Resize', img_scale=[(824, 824), (1224, 1224)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0),   # the line must exist!!!
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'valid.json',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'valid.json',
        img_prefix=data_root + 'train/',
        pipeline=test_pipeline)
)
# evaluation = dict(interval=1, metric='bbox')
# runtime
# lr = 0.02 / 8 x num_gpus x img_per_gpu / 2
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[16, 19])
total_epochs = 20

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
# work_dir = './projects/work_dirs/cascade101_dconv'
work_dir = '/var/www/nextcloud/data/dbc2017/files/work_dirs/cascade101_dconv'
load_from = ""
# https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/cascade_rcnn_x101_64x4d_fpn_2x_20181218-5add321e.pth
resume_from = None
workflow = [('train', 1), ('val', 1)]
