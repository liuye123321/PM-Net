_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/revide_256x768.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/home/peishunda/liuye/code/SegFormer-master/pretrained/mit_b4.pth',
    backbone=dict(
        type='mit_b4',
        style='pytorch'),
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=768),
        loss_dehaze=dict(type='L1_Loss', loss_weight=1.0),
        loss_depth=dict(type='L1_Loss', loss_weight=0.2),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    # test_cfg=dict(mode='slide', crop_size=(1024,1024), stride=(768,768)))

# data
data = dict(samples_per_gpu=1)

# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.00001, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


evaluation = dict(interval=1, metric='mIoU')
log_config = dict(interval=50)