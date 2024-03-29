_base_ = '../_base_/default_runtime.py'

# model settings
img_size = 550
model = dict(
    type='YOLACT',
    backbone=dict(
        type='MobileNetV2',
        out_indices=(3, 4, 5, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    # backbone1 for predicted only.
    backbone1=dict(
        type='MobileNetV2',
        out_indices=(3, 4, 5, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    # backbone_pred for joining to segm
    backbone_pred=dict(
        type='MobileNetV2',
        out_indices=(3, 4, 5, 6),
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        init_cfg=dict(
            type='Pretrained', checkpoint='open-mmlab://mmdet/mobilenet_v2')),
    neck=dict(
        type='FPN',
        in_channels=[16, 32, 32, 32],
        out_channels=32,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        upsample_cfg=dict(mode='bilinear')),
    bbox_head=dict(
        type='YOLACTHead',
        num_classes=1,
        in_channels=32,
        feat_channels=32,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=3,
            scales_per_octave=1,
            base_sizes=[16, 32, 32, 64, 128],
            ratios=[0.5, 1.0, 2.0],
            strides=[550.0 / x for x in [35, 18, 18, 9, 5]],
            centers=[(550 * 0.5 / x, 550 * 0.5 / x)
                     for x in [35, 18, 18, 9, 5]]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            reduction='none',
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.5),
        num_head_convs=1,
        num_protos=32,
        use_ohem=True),
    mask_head=dict(
        type='YOLACTProtonet',
        in_channels=32,
        num_protos=32,
        num_classes=1,
        max_masks_to_train=100,
        loss_mask_weight=6.125),
    segm_head=dict(
        type='YOLACTSegmHead',
        num_classes=1,
        in_channels=32,
        loss_segm=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0.,
            ignore_iof_thr=-1,
            gt_max_assign_all=False),
        # smoothl1_beta=1.,
        allowed_border=-1,
        pos_weight=-1,

        neg_pos_ratio=3,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        iou_thr=0.5,
        top_k=200,
        max_per_img=100))
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/xulinquan/code/mmdetection/data/20220315/'
img_norm_cfg = dict(
    mean=[123.68, 116.78, 103.94], std=[58.40, 57.12, 57.38], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='FilterAnnotations', min_gt_bbox_wh=(4.0, 4.0)),
    # dict(
    #   type='Expand',
    #   mean=img_norm_cfg['mean'],
    #   to_rgb=img_norm_cfg['to_rgb'],
    #   ratio_range=(1, 4)),
    # dict(
    #    type='MinIoURandomCrop',
    #    min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
    #    min_crop_size=0.3),
    dict(type='Resize', img_scale=(img_size, img_size), keep_ratio=False),
    # dict(type='RandomFlip', flip_ratio=0.5),
    # dict(
    #    type='PhotoMetricDistortion',
    #    brightness_delta=32,
    #    contrast_range=(0.5, 1.5),
    #    saturation_range=(0.5, 1.5),
    #    hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(img_size, img_size),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'gt_masks']),
            #dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
        ])
]
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + '0315_train.json',
        img_prefix=data_root + 'train_0727/',
        #ann_file=data_root + 'val_test.json',
        #img_prefix=data_root + 'val_0727/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + '0315_test.json',
        img_prefix=data_root + 'val_0727/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + '0315_test.json',
        img_prefix=data_root + 'val_0727/',
        pipeline=test_pipeline))
# optimizer
# optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=5e-4)

optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999))

optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    step=[12, 18, 20, 22])
runner = dict(type='EpochBasedRunner', max_epochs=30)
cudnn_benchmark = True
evaluation = dict(metric=['segm'])
