_base_ = [
    '../../_base_/models/segformer.py',
    '../../_base_/datasets/revide_768x768.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='/mnt/hard_disk/peishunda/video_dehaze/SegFormer-master-multiframe-depth_v3/pretrained/mit_b0.pth',
    backbone=[
        dict(type='mit_b0', style='pytorch'),
        dict(type='mit_b0_p', style='pytorch')
    ],
    decode_head=dict(
        type='SegFormerHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=3,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=384),
        loss_dehaze=dict(type='L1_Loss', loss_weight=1.0),
        loss_depth=dict(type='L1_Loss', loss_weight=0.2),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    # test_cfg=dict(mode='slide', test_size=[(0, 928, 0, 1376), (0, 928, 1364, 2708), (904, 1800, 0, 1376), (904, 1800, 1364, 2708)],
    #               drop_size=[12, 6])) #[h,w]
    # test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))
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


data = dict(samples_per_gpu=2)  #设置训练的batchsize
evaluation = dict(interval=1, metric='mIoU')
