# dataset_type = 'CustomNuScenesDataset'
dataset_type = 'NuScenesDataset'
dataset_root = 'data/nuscenes/'

input_modality = dict(
    use_lidar=True,
    use_camera=False,
    use_radar=False,
    use_map=False,
    use_external=False,
)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
# point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
# voxel_size = [0.2, 0.2, 8]
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
out_size_factor = 8


# Model
grid_config = {
    'x': [-51.2, 51.2, 0.8],
    'y': [-51.2, 51.2, 0.8],
    'z': [-5, 3, 8],
    'depth': [1.0, 60.0, 0.5], # original
    # 'depth': [1.0, 80.0, 0.5],
}

# arch config
embed_dims = 256
num_layers = 6
# num_query = 900
num_query = 32400
num_frames = 8
num_levels = 4
num_points = 4

model = dict(
    type='SparseBEV',
    pts_voxel_layer=dict(
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),
    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=5,
        sparse_shape=[41, 1440, 1440],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    # lidar backbone #
    pts_bbox_head=dict(
        type='SparseBEVHead',
        num_classes=10,
        in_channels=embed_dims,
        num_query=num_query,
        query_denoising=False,
        query_denoising_groups=10,
        code_size=10,
        code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        pc_range=point_cloud_range,
        sync_cls_avg_factor=True,
        transformer=dict(
            type='SparseBEVTransformer',
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_points=num_points,
            num_layers=num_layers,
            num_levels=num_levels,
            num_classes=10,
            code_size=10,
            pc_range=point_cloud_range),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            score_threshold=0.05,
            num_classes=10),
        # bbox_coder=dict(
        #     type='TransFusionBBoxCoder',
        #     pc_range=point_cloud_range[:2],
        #     voxel_size=voxel_size[:2],
        #     out_size_factor=out_size_factor,
        #     post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
        #     score_threshold=0.0,
        #     code_size=10,
        # ),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=embed_dims // 2,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
        # loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=0.15),
        # loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        # loss_iou=dict(type='GIoULoss', reduction='mean', loss_weight=0.25)),  # 추가된 부분
        # loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=0.25)),
    train_cfg=dict(pts=dict(
        grid_size=[1440, 1440, 40],
        # grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=out_size_factor,
        # out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0),
        )
    )),
    # train_cfg=dict(pts=dict(
    #     assigner=dict(
    #         type='HungarianAssigner3D',
    #         # iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
    #         cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
    #         reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25 ,pc_range = point_cloud_range),
    #         iou_cost=dict(type='IoU3DCost', weight=0.25)
    #         ),
    #     pos_weight=-1,
    #     gaussian_overlap=0.1,
    #     min_radius=2,
    #     grid_size=[1440, 1440, 40],  # [x_len, y_len, 1]
    #     voxel_size= voxel_size,
    #     out_size_factor= out_size_factor,
    #     code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
    #     point_cloud_range= point_cloud_range,
    # )),
    
)

file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR',load_dim=5,use_dim=[0,1,2,3,4],file_client_args=file_client_args),
    dict(type='LoadPointsFromMultiSweeps_moon',sweeps_num=10,use_dim=[0, 1, 2, 3, 4],),
    # dict(type='PointToMultiViewDepth_moon', downsample=1, grid_config=grid_config),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root = dataset_root,
            info_path= dataset_root+ 'nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ))),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d' ,'points'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp'))
]

test_pipeline = [
    dict(type='LoadPointsFromFile',coord_type='LIDAR',load_dim=5,use_dim=[0,1,2,3,4],file_client_args=file_client_args),
    dict(type='LoadPointsFromMultiSweeps_moon',sweeps_num=10,use_dim=[0, 1, 2, 3, 4],),
    # dict(type='PointToMultiViewDepth_moon', downsample=1, grid_config=grid_config),
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ObjectSample',
        db_sampler=dict(
            data_root = dataset_root,
            info_path= dataset_root+ 'nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ))),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points'], meta_keys=(
        'filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp'))
]

# data = dict(
#     samples_per_gpu=1,
#     workers_per_gpu=8,
#     train=dict(
#         type='CBGSDataset',
#         dataset=dict(
#             type=dataset_type,
#             data_root=dataset_root,
#             ann_file=dataset_root + '/nuscenes_infos_train.pkl',
#             load_interval=1,
#             pipeline=train_pipeline,
#             classes=class_names,
#             modality=input_modality,
#             test_mode=False,
#             box_type_3d='LiDAR')),
#     val=dict(
#         type=dataset_type,
#         data_root=dataset_root,
#         ann_file=dataset_root + '/nuscenes_infos_val.pkl',
#         load_interval=1,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'),
#     test=dict(
#         type=dataset_type,
#         data_root=dataset_root,
#         ann_file=dataset_root + '/nuscenes_infos_val.pkl',
#         load_interval=1,
#         pipeline=test_pipeline,
#         classes=class_names,
#         modality=input_modality,
#         test_mode=True,
#         box_type_3d='LiDAR'))

data = dict(
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_train_radar.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_radar.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        ann_file=dataset_root + 'nuscenes_infos_val_radar.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    # lr=0.0001,
    paramwise_cfg=dict(custom_keys={
        'img_backbone': dict(lr_mult=0.1),
        'sampling_offset': dict(lr_mult=0.1),
    }),
    weight_decay=0.01
)

optimizer_config = dict(
    type='Fp16OptimizerHook',
    loss_scale=512.0,
    grad_clip=dict(max_norm=35, norm_type=2)
)

# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

# # for transfusion leaning policy
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 0.0001),
#     cyclic_times=1,
#     step_ratio_up=0.4)
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.8947368421052632, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4)

total_epochs = 24
batch_size = 2

# load pretrained weights
# load_from = 'pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
load_from = None
# revise_keys = [('backbone', 'img_backbone')]

# resume the last training
# resume_from = 'outputs/SparseBEV/2024-08-01/10-00-26/latest.pth'
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=1,
    hooks=[
        dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
        dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
    ]
)

# evaluation
eval_config = dict(interval=total_epochs)

# other flags
debug = True
