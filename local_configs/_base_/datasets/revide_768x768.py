_base_ = './revide.py'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (768, 768)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', img_scale=(2708, 1800), ratio_range=(0.5, 1.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', flip_ratio=0.5),
    # dict(type='PhotoMetricDistortion'),
    dict(type='ShowImage', show=False),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'ref', 'gt', 'refgt']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2708, 1800),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='TestCrop', crop_size=(1792, 2688)),
            # dict(type='TestPad', size=(1800, 2712), padding_mode='edge'),
            # dict(type='TestPad', padding=(0, 0, 4, 0), padding_mode='edge'),  #'constant', 'edge', 'reflect', symmetric'
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'ref', 'gt', 'refgt']),
            dict(type='Collect', keys=['img', 'ref', 'gt', 'refgt']),
        ])
]
data = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))