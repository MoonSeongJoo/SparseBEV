import math
import torch
import torch.nn as nn
from mmcv.runner import force_fp32
from mmdet.core import multi_apply, reduce_mean
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from .bbox.utils import normalize_bbox, encode_bbox
from .utils import VERSION
import torch.nn.functional as F
import open3d as o3d
import numpy as np
import spconv.pytorch as spconv


class TNet(nn.Module):
    def __init__(self, k):
        super(TNet, self).__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.InstanceNorm1d(512)
        self.bn5 = nn.InstanceNorm1d(256)

    def forward(self, x):
        B, D, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2)
        x = x.view(B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        iden = torch.eye(self.k, requires_grad=True).repeat(B, 1, 1).to(x.device)
        x = x.view(-1, self.k, self.k) + iden
        return x

class PointNet_reduced(nn.Module):
    
    def __init__(self, num_queries=900):
        super(PointNet_reduced, self).__init__()
        self.tnet1 = TNet(k=3)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, num_queries, 1)
        self.tnet2 = TNet(k=64)
        self.num_queries = num_queries
            
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(num_queries)
    
    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1)  # [B, C, N]
        
        # Input transform
        trans = self.tnet1(x)
        x = torch.bmm(trans, x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transform
        trans_feat = self.tnet2(x)
        x = torch.bmm(trans_feat, x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x, _ = torch.max(x, 2)
        x = x.view(B, self.num_queries)
        
        return x

# Voxelization layer - Point cloud to Voxel
class VoxelizationLayer:
    def __init__(self, voxel_size, grid_size, max_num_points_per_voxel):
        self.voxel_size = voxel_size
        self.grid_size = grid_size
        self.max_num_points_per_voxel = max_num_points_per_voxel

    def forward(self, point_cloud):
        # point_cloud: (batch, num_points, 3), torch.tensor type
        batch_size = point_cloud.size(0)
        
        voxel_coords_list = []
        voxel_points_list = []

        for b in range(batch_size):
            # Convert point cloud to numpy for Open3D compatibility
            points = point_cloud[b].cpu().numpy()

            # Create an Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            # Apply voxelization using Open3D
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=self.voxel_size)

            # Store voxel information: coordinates and voxelized points
            voxels = voxel_grid.get_voxels()

            voxel_coords = []
            voxel_points = []

            for voxel in voxels:
                voxel_coords.append(voxel.grid_index)
                voxel_points.append(voxel_grid.get_voxel_center_coordinate(voxel.grid_index))

            # Convert to numpy arrays
            voxel_coords = np.array(voxel_coords)
            voxel_points = np.array(voxel_points)

            # Convert back to tensor and send to appropriate device
            voxel_coords = torch.tensor(voxel_coords, dtype=torch.long).to(point_cloud.device)
            voxel_points = torch.tensor(voxel_points, dtype=torch.float32).to(point_cloud.device)

            # Save per-batch voxelized data
            voxel_coords_list.append(voxel_coords)
            voxel_points_list.append(voxel_points)

        # Stack the list of tensors along the batch dimension
        voxel_coords_stack = torch.stack(voxel_coords_list, dim=0)
        voxel_points_stack = torch.stack(voxel_points_list, dim=0)

        return voxel_coords_stack, voxel_points_stack


# Voxel Feature Encoding (VFE) layer
class VFE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VFE, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, voxel_points):
        """
        voxel_coords: (B, num_voxels, 3), each voxel's grid index (x, y, z)
        voxel_points: (B, num_voxels, C), each voxel's center coordinate features
        """
        B, num_voxels, C = voxel_points.size()

        # Apply fully connected layer to the voxel features (voxel_points)
        x = voxel_points.view(-1, C)  # Flatten the voxels to apply the FC layer
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)

        # Reshape back to (B, num_voxels, output_dim)
        x = x.view(B, num_voxels, -1)

        # Here you might want to aggregate the features within each voxel grid cell
        # since the original version does max-pooling along the P dimension.
        # If `voxel_coords` is needed for pooling or further processing, 
        # that step would be added here.

        return x  # Output is (B, num_voxels, output_dim)


# 3D Convolutional Middle Layers
class MiddleLayers3D(nn.Module):
    def __init__(self):
        super(MiddleLayers3D, self).__init__()
        self.conv1 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x
    
class SparseConvNet(nn.Module):
    def __init__(self):
        super(SparseConvNet, self).__init__()
        self.sparse_conv1 = spconv.SparseConv3d(128, 64, kernel_size=3, padding=1)
        # self.sparse_bn1 = spconv.SparseBatchNorm3d(64)
        # self.sparse_bn1 = spconv.SparseBatchNorm(64)
        self.sparse_bn1 = nn.BatchNorm3d(64)
        self.sparse_conv2 = spconv.SparseConv3d(64, 64, kernel_size=3, padding=1)
        # self.sparse_bn2 = spconv.SparseBatchNorm3d(64)
        # self.sparse_bn2 = spconv.SparseBatchNorm(64)
        self.sparse_bn2 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU()
        self.final_conv = nn.Conv3d(64, 64, kernel_size=3, padding=1)  # depth, height, width 조정을 위한 추가 layer

    def forward(self, voxel_features, voxel_coords, spatial_shape, batch_size):
        # voxel_features: (N, C) tensor
        # voxel_coords: (N, 3) tensor
        
        # Convert voxel features and coordinates to sparse tensor
        voxel_tensor = spconv.SparseConvTensor(features=voxel_features, indices=voxel_coords , spatial_shape=spatial_shape, batch_size=batch_size)
        # x = self.relu(self.sparse_bn1(self.sparse_conv1(voxel_tensor)))
        # x = self.relu(self.sparse_bn2(self.sparse_conv2(voxel_tensor_out)))
        
        sparse_out = self.sparse_conv1(voxel_tensor)
        batch_size = voxel_tensor.batch_size
        num_voxels = sparse_out.features.size(0)
        dense_features = sparse_out.features.view(batch_size, 64, 1, 1, num_voxels)  # 4D 텐서로 변환
        # dense_features = sparse_out.features.view(-1,64) # (N, C) 형태로 변환
        x = self.relu(self.sparse_bn1(dense_features))
        
        voxel_tensor_out = spconv.SparseConvTensor(features=x.view(-1, 64), indices=sparse_out.indices , spatial_shape=voxel_tensor.spatial_shape, batch_size=batch_size)
        assert not torch.isnan(voxel_features).any(), "voxel_features contains nan"
        sparse_out2 = self.sparse_conv2(voxel_tensor_out)
        num_voxels2 = sparse_out2.features.size(0)
        dense_features2 = sparse_out2.features.view(batch_size, 64, 1, 1, num_voxels2) # 4D 텐서로 변환
        # dense_features2 = sparse_out2.features.view(-1,64) # (N, C) 형태로 변환
        x = self.relu(self.sparse_bn2(dense_features2))
        
        # 추가 Conv를 통해 원하는 크기로 조정
        x = x.view(batch_size, 64, 1, 1, num_voxels2)  # 임시로 5D 텐서로 변환
        x = self.final_conv(x)  # (batch_size, 64, depth, height, width)

        # 최종적으로 depth=30, height=30, width=30로 조정
        x = F.interpolate(x, size=(30, 30, 30), mode='trilinear', align_corners=False)
        
        return x

# Region Proposal Network (RPN)
class RPN(nn.Module):
    def __init__(self, input_dim, num_anchors):
        super(RPN, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()

        self.conv_cls = nn.Conv2d(128, 256, kernel_size=1)
        # self.conv_reg = nn.Conv2d(128, 10, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        cls_logits = self.conv_cls(x)  # Classification output
        # reg_output = self.conv_reg(x)  # Regression output (bounding boxes)
        # return cls_logits , reg_output
        return cls_logits


# VoxelNet model
class VoxelNet(nn.Module):
    def __init__(self, voxel_size, grid_size, max_num_points_per_voxel, num_anchors):
        super(VoxelNet, self).__init__()
        self.voxelization = VoxelizationLayer(voxel_size, grid_size, max_num_points_per_voxel)
        self.vfe = VFE(input_dim=3, output_dim=128)
        self.sparseconv = SparseConvNet()
        self.middle_layers = MiddleLayers3D()
        self.rpn = RPN(input_dim=1920, num_anchors=num_anchors)

    def forward(self, point_cloud):
        voxel_coords,voxel_points = self.voxelization.forward(point_cloud)
        voxel_features = self.vfe(voxel_points)
        
        # Change voxel_features shape to (N, C) for spconv
        B, num_voxels, C = voxel_features.size()
        voxel_features = voxel_features.view(B * num_voxels, C)
        
        # Change voxel_coords shape to (N, 3) for spconv
        voxel_coords = voxel_coords.view(B* num_voxels, 3)

        # 배치 인덱스 추가
        batch_indices = torch.arange(B).repeat_interleave(num_voxels).view(-1, 1).cuda()
        voxel_coords = torch.cat((batch_indices, voxel_coords), dim=1)
        # voxel_coords를 torch.int32로 캐스팅
        voxel_coords = voxel_coords.to(torch.int32)
        
        # Apply SparseConvNet
        sparse_features = self.sparseconv(voxel_features, voxel_coords , tuple(self.voxelization.grid_size) ,B)
        
        # Reshape sparse_features back to the format for MiddleLayers3D
        # Assuming the output of spconv is in (B, C, D, H, W)
        # sparse_features = sparse_features.dense()  # Convert back to dense tensor
        # sparse_features = sparse_features.permute(0, 4, 1, 2, 3)  # (B, C, D, H, W)
        
        # Process with middle layers and RPN
        middle_features = self.middle_layers(sparse_features)
        middle_features = middle_features.view(B , -1, 30, 30)
        # middle_features = torch.squeeze(middle_features, dim=2)  # Squeeze depth dimension to use 2D conv in RPN
        # cls_logits , reg_logits = self.rpn(middle_features)
        cls_logits = self.rpn(middle_features)
        
        # return cls_logits , reg_logits
        return cls_logits 


@HEADS.register_module()
class SparseBEVHead(DETRHead):
    def __init__(self,
                 *args,
                 num_classes,
                 in_channels,
                 query_denoising=True,
                 query_denoising_groups=10,
                 bbox_coder=None,
                 code_size=10,
                 code_weights=[1.0] * 10,
                 train_cfg=dict(),
                 test_cfg=dict(max_per_img=100),
                 **kwargs):
        self.code_size = code_size
        self.code_weights = code_weights
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = in_channels

        super(SparseBEVHead, self).__init__(num_classes, in_channels, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights), requires_grad=False)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        # self.pointNet_contents = PointNet(num_queries=900 , is_contents=True)
        # self.pointNet_contents = PointNet_reduced(num_queries=900)
        # self.pointNet_position = PointNet(num_queries=900 , is_contents=False)
        self.voxelnet = VoxelNet(voxel_size = 0.2 , grid_size = [1440, 1440, 41], max_num_points_per_voxel = 35, num_anchors = 5)
        # self.fc = nn.Linear(900,self.num_query*self.embed_dims)
        # self.fc1 = nn.Linear(900,self.num_query*self.embed_dims)
        
        self.dn_enabled = query_denoising
        self.dn_group_num = query_denoising_groups
        self.dn_weight = 1.0
        self.dn_bbox_noise_scale = 0.5
        self.dn_label_noise_scale = 0.5

    def _init_layers(self):
        self.init_query_bbox = nn.Embedding(self.num_query, 10)  # (x, y, z, w, l, h, sin, cos, vx, vy)
        self.label_enc = nn.Embedding(self.num_classes + 1, self.embed_dims - 1)  # DAB-DETR
        # self.fc = nn.Linear(self.embed_dims*2,self.embed_dims) # DDP error : 안쓰여지는 network
        
        nn.init.zeros_(self.init_query_bbox.weight[:, 2:3])
        nn.init.zeros_(self.init_query_bbox.weight[:, 8:10])
        nn.init.constant_(self.init_query_bbox.weight[:, 5:6], 1.5)

        self.grid_size = int(math.sqrt(self.num_query))
        assert self.grid_size * self.grid_size == self.num_query
        x = y = torch.arange(self.grid_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')  # [0, grid_size - 1]
        xy = torch.cat([xx[..., None], yy[..., None]], dim=-1)
        xy = (xy + 0.5) / self.grid_size  # [0.5, grid_size - 0.5] / grid_size ~= (0, 1)
        with torch.no_grad():
            self.init_query_bbox.weight[:, :2] = xy.reshape(-1, 2)  # [Q, 2]
            # self.grid = xy.reshape(-1, 2)

    def init_weights(self):
        self.transformer.init_weights()
    
    # def lidar_to_bev(self, global_points, resolution=100, bev_size=(30, 30), z_limit=(-10.0, 20.0)):
    #     """
    #     글로벌 좌표계의 Lidar 포인트를 BEV 형식으로 변환합니다.

    #     :param global_points: 글로벌 좌표계의 포인트 (BxN tensor, [x, y, z])
    #     :param resolution: BEV 해상도
    #     :param bev_size: BEV 크기 (height, width) 튜플
    #     :param z_limit: Z축 제한 (min, max)
    #     :return: BEV 텐서 (B x H x W)
    #     """
    #     # BEV 텐서 초기화
    #     bev = torch.zeros((global_points.size(0), bev_size[0], bev_size[1],3))

    #     for b in range(global_points.size(0)):  # 배치 크기만큼 반복
    #         x = global_points[b, :, 0]
    #         y = global_points[b, :, 1]
    #         z = global_points[b, :, 2]

    #         # x, y 좌표를 BEV 픽셀 좌표로 변환
    #         x_pixel = ((x / resolution) + (bev_size[1] / 2)).long()
    #         y_pixel = ((y / resolution) + (bev_size[0] / 2)).long()

    #         # Z축 제한 적용
    #         z_mask = (z >= z_limit[0]) & (z <= z_limit[1])
    #         x_pixel = x_pixel[z_mask]
    #         y_pixel = y_pixel[z_mask]
    #         z = z[z_mask]

    #         # BEV 텐서에 Z값을 할당
    #         for i in range(len(x_pixel)):
    #             if 0 <= x_pixel[i] < bev_size[1] and 0 <= y_pixel[i] < bev_size[0]:
    #                 # bev[b, y_pixel[i], x_pixel[i]] = z[i]
    #                 # x, y, z 값을 각각의 채널에 할당
    #                 bev[b, y_pixel[i], x_pixel[i], 0] = x[i]  # x값
    #                 bev[b, y_pixel[i], x_pixel[i], 1] = y[i]  # y값
    #                 bev[b, y_pixel[i], x_pixel[i], 2] = z[i]  # z값

    #     return bev
    # def map_global_points_to_bev(self, global_points, bev_grid, grid_size=(30, 30)):
    #     """
    #     global_points를 bev_grid에 매핑합니다.

    #     :param global_points: shape [B, N, 3]의 텐서 (x, y, z 좌표)
    #     :param bev_grid: shape [900, 2]의 텐서 (x, y 좌표)
    #     :param grid_size: BEV 그리드 크기 (height, width)
    #     :return: 매핑된 포인트들
    #     """
    #     B, N, _ = global_points.shape
    #     # bev_grid를 [-1, 1] 범위로 변환
    #     bev_grid_normalized = (bev_grid / (grid_size[0] - 1)) * 2 - 1  # [0, grid_size-1] -> [-1, 1]
        
    #     # bev_grid를 (1, H, W, 2) 형태로 변환
    #     bev_grid_tensor = bev_grid_normalized.view(1, grid_size[0], grid_size[1], 2)

    #     # global_points를 (B, N, 1, 3) 형태로 변환
    #     global_points_tensor = global_points.view(B, N, 1, 3)

    #     # x, y 좌표를 사용하여 z값을 매핑합니다. 
    #     # grid_sample은 입력이 4D이어야 하므로, global_points_tensor를 4D로 변환합니다.
    #     mapped_points = F.grid_sample(global_points_tensor, bev_grid_tensor, mode='bilinear', align_corners=True)

    #     # 결과를 (B, 1, 3) 형태로 변환 후 squeeze하여 (B, 3)으로 만듭니다.
    #     mapped_points = mapped_points.squeeze(2)  # (B, N, 3) -> (B, 3)
        
    #     return mapped_points
    def query_sync(self, pc_feature, grid):
        # pc_feature를 (B, C, 1, N) 형태로 변환
        B, N = pc_feature.shape
        pc_feature = pc_feature.view(B,1,30,30)  # (B, C) -> (B, C, 1)

        # grid를 (B, N, 1, 2) 형태로 변환
        grid = grid.view(B, 30, 30, 2)  # (B, N, 2) -> (B, N, 1, 2)

        # # grid의 값을 [-1, 1] 범위로 정규화
        # grid = grid * 2 - 1  # [0, 1] -> [-1, 1]
        # 정규화 과정
        min_val = torch.min(grid)
        max_val = torch.max(grid)

        # [-1, 1] 사이로 정규화
        grid_normalized = 2 * ((grid - min_val) / (max_val - min_val)) - 1

        # grid_sample을 사용하여 pc_feature 맵핑
        output = F.grid_sample(pc_feature, grid_normalized, mode='bilinear', padding_mode='border', align_corners=True)

        return output.view(B,-1)
    
    # def query_sync_1 (self, pc_feature, grid):
        
    #     B,N,C =pc_feature.shape
    #     # pc_feature를 (N, C, H, W) 형태로 변환
    #     pc_feature = pc_feature.permute(0, 2, 1).unsqueeze(3)  # (Batch, C, H, W) -> (1, 256, 900, 1)

    #     # grid의 값을 [-1, 1] 범위로 정규화
    #     grid = grid * 2 - 1  # [0, 1] -> [-1, 1]로 변환

    #     # grid를 (B, 900, 1, 2) 형태로 변환
    #     grid = grid.view(B, N, 1, 2)  # (B, 900, 1, 2)

    #     # grid_sample을 사용하여 pc_feature 맵핑
    #     output = F.grid_sample(pc_feature, grid, mode='bilinear', padding_mode='border') # (B,C,H,W)
    #     return output.view(B,C,-1).permute(0,2,1)
    
    # def calc_bbox_position (self,query_pc_position):
        
    #     normalized_query_pc_position = query_pc_position.clone()
    #     # x, y 좌표에 대해 개별적으로 정규화
    #     min_val = query_pc_position[:,:,0].min()
    #     max_val = query_pc_position[:,:,0].max()
    #     normalized_query_pc_position[:,:,0] = (query_pc_position[:,:,0] - min_val) / (max_val - min_val)
    #     min_val = query_pc_position[:,:,1].min()
    #     max_val = query_pc_position[:,:,1].max()
    #     normalized_query_pc_position[:,:,1] = (query_pc_position[:,:,1] - min_val) / (max_val - min_val)
        
    #     return normalized_query_pc_position
    
    def calc_normalize (self, query_pc_contents):
        
        min_val = query_pc_contents.min()
        max_val = query_pc_contents.max()
        normalized_query_pc = ((query_pc_contents - min_val) / (max_val - min_val)) * 10
        # 정수형으로 변환 (0~11 범위의 정수)
        normalized_query_pc = normalized_query_pc.round().long()

        return normalized_query_pc

    def forward(self, mlvl_feats, img_metas, points_raw, points_gt, points_mis,global_points,z_points):
        query_bbox_clone = self.init_query_bbox.weight.clone()  # [Q, 10]
        # grid = query_bbox_raw[:,:2]
        # query_bbox_raw[:,:2] = global_points.squeeze(0)
        # query_bbox_raw[:,2] = z_points.squeeze(0)
        #query_bbox[..., :3] = query_bbox[..., :3].sigmoid()
        # query denoising
        B = mlvl_feats[0].shape[0]

        # reduce_point_raw , _ = self.pointNet.farthest_point_sampling(points_raw,30000)
        # query_position_pc = self.pointNet_position(points_raw)
        # query_pc_position = self.calc_bbox_position(query_position_pc)
        # glob_feat = self.pointNet_contents(points_raw)
        voxel_cls_feat = self.voxelnet(points_raw)
        voxel_cls_feat = voxel_cls_feat.view(B,256,-1).permute(0,2,1)  # query feature
        # voxel_reg_feat = voxel_reg_feat.view(B,10,-1).permute(0,2,1)   # query position embedding
        # grid = query_pc_position[:,:,:2]
        # emb_glob_feat = self.init_query_bbox(glob_feat.squeeze(0).round().long())
        # emb_grid = emb_glob_feat[:,:2]
        # query_pc_contents = self.query_sync(glob_feat,emb_grid) # (B,N,C)
        # mid_query_pc = self.query_sync_1(pc_cls,grid)
        # mid_query_pc = self.fc(query_pc_contents)
        # normalized_query_pc = self.calc_normalize(query_pc_contents)
        
        # emb_query_pc_contents = self.label_enc(query_pc_contents.squeeze(2).long())
        # emb_query_pc_contents = self.label_enc(normalized_query_pc.reshape(-1))
        # query_bbox_raw, query_feat_raw, attn_mask, mask_dict = self.prepare_for_dn_input(B, voxel_reg_feat , img_metas, voxel_cls_feat)
        query_bbox_raw, query_feat_raw, attn_mask, mask_dict = self.prepare_for_dn_input(B, query_bbox_clone , self.label_enc, img_metas, voxel_cls_feat)
        # query_bbox_raw, query_feat_raw, attn_mask, mask_dict = self.prepare_for_dn_input(B, query_pc_position, self.label_enc, img_metas,query_pc_contents)
        # query_bbox_raw, query_feat_raw, attn_mask, mask_dict = self.prepare_for_dn_input(B, query_bbox_raw,img_metas,query_pc_contents)
        # query_bbox_raw, query_feat_raw, attn_mask, mask_dict = self.prepare_for_dn_input(B, query_bbox_raw, self.label_enc, img_metas)
        
        query_feat = query_feat_raw 
        query_bbox = query_bbox_raw 
        # attn_mask = None
        # mask_dict = None
        
        cls_scores, bbox_preds = self.transformer(
            query_bbox,
            query_feat,
            mlvl_feats,
            attn_mask=attn_mask,
            img_metas=img_metas,
        )
        # cls_scores, bbox_preds = self.transformer(
        #     query_bbox,
        #     query_feat,
        #     mlvl_feats,
        #     img_metas=img_metas,
        # )

        bbox_preds[..., 0] = bbox_preds[..., 0] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        bbox_preds[..., 1] = bbox_preds[..., 1] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        bbox_preds[..., 2] = bbox_preds[..., 2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        bbox_preds = torch.cat([
            bbox_preds[..., 0:2],
            bbox_preds[..., 3:5],
            bbox_preds[..., 2:3],
            bbox_preds[..., 5:10],
        ], dim=-1)  # [cx, cy, w, l, cz, h, sin, cos, vx, vy]

        if mask_dict is not None and mask_dict['pad_size'] > 0:  # if using query denoising
            output_known_cls_scores = cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_bbox_preds = bbox_preds[:, :, :mask_dict['pad_size'], :]
            output_cls_scores = cls_scores[:, :, mask_dict['pad_size']:, :]
            output_bbox_preds = bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes'] = (output_known_cls_scores, output_known_bbox_preds)
            outs = {
                'all_cls_scores': output_cls_scores,
                'all_bbox_preds': output_bbox_preds,
                'enc_cls_scores': None,
                'enc_bbox_preds': None, 
                'dn_mask_dict': mask_dict,
            }
        else:
            outs = {
                'all_cls_scores': cls_scores,
                'all_bbox_preds': bbox_preds,
                'enc_cls_scores': None,
                'enc_bbox_preds': None, 
            }
            
        # outs = {
        #         'all_cls_scores': cls_scores,
        #         'all_bbox_preds': bbox_preds,
        #         'enc_cls_scores': None,
        #         'enc_bbox_preds': None, 
        #     }

        return outs
    

    def prepare_for_dn_input(self, batch_size, init_query_bbox, label_enc, img_metas, voxel_cls_feat):
    # def prepare_for_dn_input(self, batch_size, init_query_bbox,img_metas, voxel_cls_feat):
    # def prepare_for_dn_input(self, batch_size, init_query_bbox, img_metas, query_feat_pc):
    # def prepare_for_dn_input(self, batch_size, init_query_bbox, label_enc, img_metas):
        # mostly borrowed from:
        #  - https://github.com/IDEA-Research/DN-DETR/blob/main/models/DN_DAB_DETR/dn_components.py
        #  - https://github.com/megvii-research/PETR/blob/main/projects/mmdet3d_plugin/models/dense_heads/petrv2_dnhead.py

        device = init_query_bbox.device
        # indicator0 = torch.zeros([self.num_query, 1], device=device)
        # init_query_feat_raw = label_enc.weight[self.num_classes].repeat(self.num_query, 1)
        # init_query_feat_raw = torch.cat([init_query_feat_raw, indicator0], dim=1)
        # query_feat_pc = torch.cat([query_feat_pc.unsqueeze(0), indicator0], dim=2)
        
        # init_query_feat = init_query_feat_raw
        query_bbox      = init_query_bbox
        
        if self.training and self.dn_enabled:
            targets = [{
                'bboxes': torch.cat([m['gt_bboxes_3d'].gravity_center,
                                     m['gt_bboxes_3d'].tensor[:, 3:]], dim=1).cuda(),
                'labels': m['gt_labels_3d'].cuda().long()
            } for m in img_metas]

            known = [torch.ones_like(t['labels'], device=device) for t in targets]
            known_num = [sum(k) for k in known]

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_bbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets]).clone()
            bboxes = torch.cat([t['bboxes'] for t in targets]).clone()
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)

            # add noise
            known_indice = known_indice.repeat(self.dn_group_num, 1).view(-1)
            known_labels = labels.repeat(self.dn_group_num, 1).view(-1)
            known_bid = batch_idx.repeat(self.dn_group_num, 1).view(-1)
            known_bboxs = bboxes.repeat(self.dn_group_num, 1) # 9
            known_labels_expand = known_labels.clone()
            known_bbox_expand = known_bboxs.clone()

            # noise on the box
            if self.dn_bbox_noise_scale > 0:
                wlh = known_bbox_expand[..., 3:6].clone()
                rand_prob = torch.rand_like(known_bbox_expand) * 2 - 1.0
                known_bbox_expand[..., 0:3] += torch.mul(rand_prob[..., 0:3], wlh / 2) * self.dn_bbox_noise_scale
                # known_bbox_expand[..., 3:6] += torch.mul(rand_prob[..., 3:6], wlh) * self.dn_bbox_noise_scale
                # known_bbox_expand[..., 6:7] += torch.mul(rand_prob[..., 6:7], 3.14159) * self.dn_bbox_noise_scale

            # known_bbox_expand = encode_bbox(known_bbox_expand, self.pc_range)
            known_bbox_expand = encode_bbox(known_bbox_expand.clone(), self.pc_range)  # Create a clone before encoding
            known_bbox_expand[..., 0:3].clamp_(min=0.0, max=1.0)
            # nn.init.constant(known_bbox_expand[..., 8:10], 0.0)

            # noise on the label
            if self.dn_label_noise_scale > 0:
                p = torch.rand_like(known_labels_expand.float())
                chosen_indice = torch.nonzero(p < self.dn_label_noise_scale).view(-1)  # usually half of bbox noise
                ######### original ################
                # new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                # known_labels_expand.scatter_(0, chosen_indice, new_label)

                new_label = torch.randint(0, self.num_classes, chosen_indice.size(), device=device)  # Corrected this to avoid creating labels under a view
                known_labels_expand = known_labels_expand.clone()  # Clone before modifying
                known_labels_expand.scatter_(0, chosen_indice, new_label)

            known_feat_expand = label_enc(known_labels_expand)
            # combined_input = torch.cat([query_feat_pc.reshape(-1), known_labels_expand], dim=0)
            # emb_query_pc_contents = self.fc(query_feat_pc).view(self.num_query,-1)
            # global_pc_contents = self.fc1(glob_feat).view(self.num_query,-1)
            # combined_emb = label_enc(combined_input)
            # emb_query_pc_contents, known_feat_expand = torch.split(combined_emb, 
            #                                             [query_feat_pc.numel(), 
            #                                             known_labels_expand.numel()],
            #                                             dim=0)
            indicator1 = torch.ones([known_feat_expand.shape[0], 1], device=device)  # add dn part indicator
            # indicator2 = torch.ones([emb_query_pc_contents.shape[0], 1], device=device)  # add dn part indicator
            
            # query_feat_pc_expand = torch.cat([emb_query_pc_contents, indicator2], dim=1)
            known_feat_expand = torch.cat([known_feat_expand, indicator1], dim=1)

            # construct final query
            dn_single_pad = int(max(known_num))
            dn_pad_size = int(dn_single_pad * self.dn_group_num)
            # dn_query_bbox = torch.zeros([dn_pad_size, init_query_bbox.shape[-1]], device=device)
            dn_query_bbox = torch.zeros([dn_pad_size, query_bbox.shape[-1]], device=device)
            dn_query_feat = torch.zeros([dn_pad_size, self.embed_dims], device=device)
            # input_query_bbox = torch.cat([dn_query_bbox, init_query_bbox], dim=0).repeat(batch_size, 1, 1)
            # input_query_feat_raw = torch.cat([dn_query_feat, init_query_feat], dim=0).repeat(batch_size, 1, 1)
            batch_dn_query_feat =dn_query_feat.repeat(batch_size, 1, 1)
            batch_dn_query_bbox =dn_query_bbox.repeat(batch_size, 1, 1)
            batch_query_bbox = query_bbox.repeat(batch_size, 1, 1)
            # batch_query_bbox = query_bbox
            # batch_query_feat_pc_expand = emb_query_pc_contents.repeat(batch_size, 1, 1)
            # batch_global_pc_contents = global_pc_contents.repeat(batch_size, 1, 1)

            input_query_pc = torch.cat([batch_dn_query_feat,voxel_cls_feat],dim=1)
            input_query_pc_bbox = torch.cat([batch_dn_query_bbox, batch_query_bbox],dim=1)
            # input_global_pc = torch.cat([batch_dn_query_feat, voxel_cls_feat],dim=1)
            # input_query_combined = torch.cat([input_query_feat_raw,input_query_pc],dim=2)

            # aggrated_query_feat = self.fc(input_query_combined)
            input_query_feat = input_query_pc
            # input_query_feat = input_query_feat_raw
            input_query_bbox = input_query_pc_bbox

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + dn_single_pad * i for i in range(self.dn_group_num)]).long()

            if len(known_bid):
                input_query_bbox[known_bid.long(), map_known_indice] = known_bbox_expand
                input_query_feat[(known_bid.long(), map_known_indice)] = known_feat_expand

            total_size = dn_pad_size + self.num_query
            attn_mask = torch.ones([total_size, total_size], device=device) < 0

            # match query cannot see the reconstruct
            attn_mask[dn_pad_size:, :dn_pad_size] = True
            for i in range(self.dn_group_num):
                if i == 0:
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), dn_single_pad * (i + 1):dn_pad_size] = True
                if i == self.dn_group_num - 1:
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), :dn_single_pad * i] = True
                else:
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), dn_single_pad * (i + 1):dn_pad_size] = True
                    attn_mask[dn_single_pad * i:dn_single_pad * (i + 1), :dn_single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'pad_size': dn_pad_size
            }
        else:
            # input_query_bbox = init_query_bbox.repeat(batch_size, 1, 1)
            input_query_bbox = query_bbox.repeat(batch_size, 1, 1)
            input_query_feat = voxel_cls_feat
            # emb_query_pc_contents = self.fc(query_feat_pc).view(self.num_query,-1)
            # input_query_feat = emb_query_pc_contents.repeat(batch_size, 1, 1)
            # global_pc_contents = self.fc1(glob_feat).view(self.num_query,-1)
            # input_global_pc = global_pc_contents.repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return input_query_bbox, input_query_feat, attn_mask, mask_dict

    def prepare_for_dn_loss(self, mask_dict):
        cls_scores, bbox_preds = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        num_tgt = known_indice.numel()

        if len(cls_scores) > 0:
            cls_scores = cls_scores.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            bbox_preds = bbox_preds.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)

        return known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt

    def dn_loss_single(self,
                       cls_scores,
                       bbox_preds,
                       known_bboxs,
                       known_labels,
                       num_total_pos=None):        
        # Compute the average number of gt boxes accross all gpus
        num_total_pos = cls_scores.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1.0).item()

        # cls loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        loss_cls = self.loss_cls(
            cls_scores,
            known_labels.long(),
            label_weights,
            avg_factor=num_total_pos
        )

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = self.dn_weight * torch.nan_to_num(loss_cls)
        loss_bbox = self.dn_weight * torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def calc_dn_loss(self, loss_dict, preds_dicts, num_dec_layers):
        known_labels, known_bboxs, cls_scores, bbox_preds, num_tgt = \
            self.prepare_for_dn_loss(preds_dicts['dn_mask_dict'])

        all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
        all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
        all_num_tgts_list = [num_tgt for _ in range(num_dec_layers)]

        dn_losses_cls, dn_losses_bbox = multi_apply(
            self.dn_loss_single, cls_scores, bbox_preds,
            all_known_bboxs_list, all_known_labels_list, all_num_tgts_list)

        loss_dict['loss_cls_dn'] = dn_losses_cls[-1]
        loss_dict['loss_bbox_dn'] = dn_losses_bbox[-1]

        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1], dn_losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls_dn'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox_dn'] = loss_bbox_i
            num_dec_layer += 1

        return loss_dict

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        num_bboxes = bbox_pred.size(0)

        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore, self.code_weights, True)
        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :9]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
                self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                gt_bboxes_list, gt_labels_list, gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos
        )

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return loss_cls, loss_bbox

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)
        
        # losses_cls, losses_bbox = self.loss_single(
        #     all_cls_scores, all_bbox_preds,
        #     all_gt_bboxes_list, all_gt_labels_list, 
        #     all_gt_bboxes_ignore_list)

        loss_dict = dict()
        #loss of proposal generated from encode feature map
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        if 'dn_mask_dict' in preds_dicts and preds_dicts['dn_mask_dict'] is not None:
            loss_dict = self.calc_dn_loss(loss_dict, preds_dicts, num_dec_layers)

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            if VERSION.name == 'v0.17.1':
                import copy
                w, l = copy.deepcopy(bboxes[:, 3]), copy.deepcopy(bboxes[:, 4])
                bboxes[:, 3], bboxes[:, 4] = l, w
                bboxes[:, 6] = -bboxes[:, 6] - math.pi / 2

            bboxes = LiDARInstance3DBoxes(bboxes, 9)
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
