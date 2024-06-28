import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch.nn.functional as F
import torchvision

    
def display_depth_maps(imgs, depth_map, mis_calibrated_depth_map, cid):
    # """
    # 이미지 위에 depth map과 mis_calibrated depth map을 오버레이하여 디스플레이하는 함수
    # """
    # depth_map = depth_map.to(dtype=torch.uint8).squeeze()
    # mis_calibrated_depth_map = mis_calibrated_depth_map.to(dtype=torch.uint8).squeeze()
    # depth_map_color = colormap(depth_map[cid])
    # mis_calibrated_depth_map_color = colormap(mis_calibrated_depth_map[cid])
    # # imgs는 [B, C, H, W] 형태를 가정합니다. PyTorch 텐서일 경우 .numpy()로 변환 필요
    img = imgs.squeeze()[cid].permute(1, 2, 0).cpu().detach().numpy()  # CHW -> HWC로 변경하고 numpy 배열로 변환
    # 이미지 데이터가 float 타입인 경우 0과 1 사이로 정규화
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img - img.min()) / (img.max() - img.min())

    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # # 원본 이미지 표시
    # ax[0].imshow(img)
    # # Depth map을 투명하게 오버레이
    # ax[0].imshow(depth_map_color.numpy(), cmap='magma')
    # # ax[0].scatter(depth_map_color.cpu()[0, :], depth_map.cpu()[1, :], c='r', s=1) # 포인트 클라우드 점 표시
    # ax[0].set_title('Depth Map Overlay')
    # ax[0].axis('off')

    # # 원본 이미지 표시
    # ax[1].imshow(img)
    # # Mis-Calibrated Depth map을 투명하게 오버레이
    # ax[1].imshow(mis_calibrated_depth_map_color.numpy(), cmap='magma')
    # # ax[1].scatter(mis_calibrated_depth_map_color.cpu()[0, :], mis_calibrated_depth_map.cpu()[1, :], c='r', s=1) # 포인트 클라우드 점 표시
    # ax[1].set_title('Mis-Calibrated Depth Map Overlay')
    # ax[1].axis('off')

    # plt.show()
    
    # input display
    ####### display input signal #########        
    plt.figure(figsize=(10, 10))
    plt.subplot(311)
    plt.imshow(img)
    plt.title("camera_input", fontsize=15)
    plt.axis('off')

    plt.subplot(312)
    plt.imshow(torchvision.utils.make_grid(depth_map[cid]).cpu().numpy() , cmap='magma')
    plt.title("calibrated_lidar_input", fontsize=15)
    plt.axis('off') 

    plt.subplot(313)
    plt.imshow(torchvision.utils.make_grid(mis_calibrated_depth_map[cid]).cpu().numpy() , cmap='magma')
    plt.title("mis-calibrated_lidar_input", fontsize=15)
    plt.axis('off')
    plt.savefig('raw_depth_img_comparison.png')
    plt.close()   
    print ('display end')
    # ############ end of display input signal ###################

def overlay_imgs(rgb, lidar, idx=0):
    std = [0.229, 0.224, 0.225]
    mean = [0.485, 0.456, 0.406]

    rgb = rgb.clone().cpu().permute(1,2,0).numpy()
    rgb = rgb*std+mean
    lidar = lidar.clone()
#     print('oeverlay imgs lidar shape' , lidar.shape)

    lidar[lidar == 0] = 1000.
    lidar = -lidar
    #lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = F.max_pool2d(lidar, 3, 1, 1)
    lidar = -lidar
    lidar[lidar == 1000.] = 0.

    #lidar = lidar.squeeze()
    lidar = lidar[0][0]
    lidar = (lidar*255).int().cpu().numpy()
    lidar_color = cm.jet(lidar)
    lidar_color[:, :, 3] = 0.5
    lidar_color[lidar == 0] = [0, 0, 0, 0]
    blended_img = lidar_color[:, :, :3] * (np.expand_dims(lidar_color[:, :, 3], 2)) + \
                  rgb * (1. - np.expand_dims(lidar_color[:, :, 3], 2))
    blended_img = blended_img.clip(min=0., max=1.)
#     print('blended_img shape' , np.asarray(blended_img).shape)
    # io.imshow(blended_img)
    # io.show()
    # plt.figure()
    # plt.imshow(lidar_color)
    #io.imsave(f'./IMGS/{idx:06d}.png', blended_img)
    return blended_img , rgb , lidar_color

def points2depthmap_grid(points, height, width):
    downsample =1
    grid_config = {
                    'x': [-51.2, 51.2, 0.8],
                    'y': [-51.2, 51.2, 0.8],
                    'z': [-5, 3, 8],
                    'depth': [1.0, 60.0, 0.5], # original
                    # 'depth': [1.0, 80.0, 0.5],
                }
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), device=points.device, dtype=torch.float32)
    depth_map_grid = torch.zeros((height, width ,2), device=points.device, dtype=torch.float32)
    coor = torch.round(points[:, :2] / downsample) # uv
    depth = points[:, 2] # z
    # kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
    #     coor[:, 1] >= 0) & (coor[:, 1] < height) & (
    #         depth < grid_config['depth'][1]) & (
    #             depth >= grid_config['depth'][0])
    # coor, depth = coor[kept1], depth[kept1]
    # ranks = coor[:, 0] + coor[:, 1] * width
    # sort = (ranks + depth / 100.).argsort()
    # coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    # kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    # kept2[1:] = (ranks[1:] != ranks[:-1])
    # coor_float, depth = coor[kept2], depth[kept2]
    # coor_long = coor.to(torch.long)
    # depth_map[coor_long[:, 1], coor_long[:, 0]] = depth
    # depth_map_grid[coor_long[:, 1], coor_long[:, 0], :] = coor
    return coor, depth

def points2depthmap(points, height, width ,downsample=1):
    grid_config = {
        'x': [-51.2, 51.2, 0.8],
        'y': [-51.2, 51.2, 0.8],
        'z': [-5, 3, 8],
        'depth': [1.0, 60.0, 0.5], # original
        # 'depth': [1.0, 80.0, 0.5],
    }
    height, width = height // downsample, width // downsample
    depth_map = torch.zeros((height, width), dtype=torch.float32)
    coor = torch.round(points[:, :2] / downsample)
    depth = points[:, 2]
    kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
        coor[:, 1] >= 0) & (coor[:, 1] < height) & (
            depth < grid_config['depth'][1]) & (
                depth >= grid_config['depth'][0])
    coor, depth = coor[kept1], depth[kept1]
    ranks = coor[:, 0] + coor[:, 1] * width
    sort = (ranks + depth / 100.).argsort()
    coor, depth, ranks = coor[sort], depth[sort], ranks[sort]

    kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
    kept2[1:] = (ranks[1:] != ranks[:-1])
    coor, depth = coor[kept2], depth[kept2]
    
    coor = coor.to(torch.long)
    depth_map[coor[:, 1], coor[:, 0]] = depth
    
    return depth_map, coor, depth
    # return depth_map

def add_mis_calibration(extrinsic, intrinsic, points_lidar):
    max_r = 7.5
    max_t = 0.20
    # 회전 각도 랜덤 생성 (deg 단위)
    max_angle = max_r  # 최대 회전 각도
    rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    
    # 회전 행렬 생성
    Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                [np.sin(rotz), np.cos(rotz), 0],
                [0, 0, 1]])
    Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
                [0, 1, 0],
                [-np.sin(roty), 0, np.cos(roty)]])
    Rx = np.array([[1, 0, 0],
                [0, np.cos(rotx), -np.sin(rotx)],
                [0, np.sin(rotx), np.cos(rotx)]])
    
    # 총 회전 행렬 생성
    R = np.dot(Rz, np.dot(Ry, Rx))
    
    # 이동 벡터 랜덤 생성
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    T = np.array([transl_x, transl_y, transl_z])

    # lidar2img 행렬에 mis-calibration 적용
    lidar2img_mis_calibrated = extrinsic.clone()
    lidar2img_mis_calibrated[:3, :3] = torch.tensor(R, dtype=torch.float32) @ extrinsic[:3, :3]
    lidar2img_mis_calibrated[:3, 3] += torch.tensor(T, dtype=torch.float32)

    # intrinsic 변환
    homo_intrinsic = torch.eye(4,dtype=torch.float32)
    homo_intrinsic[:3,:3] = intrinsic
    KT = homo_intrinsic.matmul(lidar2img_mis_calibrated)

    # Mis-calibrated depth map 계산
    points_img_mis_calibrated = points_lidar.tensor[:, :3].matmul(KT[:3, :3].T) + KT[:3, 3].unsqueeze(0)
    points_img_mis_calibrated = torch.cat([points_img_mis_calibrated[:, :2] / points_img_mis_calibrated[:, 2:3], points_img_mis_calibrated[:, 2:3]], 1)
    # points_img_mis_calibrated = points_img_mis_calibrated.matmul(post_rots[cid].T) + post_trans[cid:cid + 1, :]
    # mis_calibrated_depth_map, uv , z = points2depthmap(points_img_mis_calibrated, imgs.shape[2], imgs.shape[3])
    
    return points_img_mis_calibrated

def apply_random_rt_to_point_cloud(point_cloud, max_r=7.5, max_t=0.20):
    # 회전 각도 랜덤 생성 (deg 단위)
    max_angle = max_r  # 최대 회전 각도
    rotz = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    roty = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)
    rotx = np.random.uniform(-max_angle, max_angle) * (np.pi / 180.0)

    # 회전 행렬 생성
    Rz = np.array([[np.cos(rotz), -np.sin(rotz), 0],
                   [np.sin(rotz), np.cos(rotz), 0],
                   [0, 0, 1]])
    Ry = np.array([[np.cos(roty), 0, np.sin(roty)],
                   [0, 1, 0],
                   [-np.sin(roty), 0, np.cos(roty)]])
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rotx), -np.sin(rotx)],
                   [0, np.sin(rotx), np.cos(rotx)]])

    # 총 회전 행렬 생성
    R = np.dot(Rz, np.dot(Ry, Rx))

    # 이동 벡터 랜덤 생성
    transl_x = np.random.uniform(-max_t, max_t)
    transl_y = np.random.uniform(-max_t, max_t)
    transl_z = np.random.uniform(-max_t, max_t)
    T = np.array([transl_x, transl_y, transl_z])
    
    # 포인트 클라우드에 회전 및 이동 변환 적용
    rotated_points = np.dot(point_cloud, R.T)
    transformed_points = rotated_points + T

    return transformed_points

def project_points_to_image(points, camera_intrinsics, camera_extrinsics, image_shape):
    """
    포인트 클라우드를 카메라 이미지에 투영하는 함수
    points: (N, 3) 크기의 포인트 클라우드 텐서
    camera_intrinsics: (3, 3) 크기의 카메라 내부 행렬
    camera_extrinsics: (4, 4) 크기의 카메라 외부 행렬
    image_shape: (height, width) 이미지 크기
    """
    # # 포인트 클라우드에 동차 좌표 추가
    # ones = torch.ones((points.shape[0], 1), dtype=torch.float32)
    # points_homogeneous = torch.cat([points, ones], dim=1)  # (N, 4)

    # # 카메라 좌표계로 변환
    # points_camera = (camera_extrinsics @ points_homogeneous.T).T[:, :3]

    # 이미지 평면으로 투영
    points_image = (camera_intrinsics @ points.T).T

    # 동차 좌표에서 유클리드 좌표로 변환
    uv = points_image[:, :2] / points_image[:, 2:3]
    z = points_image[:, 2]  # z 값 추출

    # 이미지 범위 내의 포인트만 필터링
    mask = (uv[:, 0] >= 0) & (uv[:, 0] < image_shape[1]) & \
           (uv[:, 1] >= 0) & (uv[:, 1] < image_shape[0])
    
    points_image = points_image[mask]
    uv = uv[mask]
    z  = z[mask]
    
    return points_image ,uv ,z

def dense_map_gpu_optimized(Pts, n, m, grid):
    device = Pts.device
    ng = 2 * grid + 1
    epsilon = 1e-8  # 작은 값 추가하여 0으로 나누는 상황 방지
    # import pdb; pdb.set_trace()
    # 초기 텐서를 GPU로 이동
    mX = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mY = torch.full((m, n), float('inf'), dtype=torch.float32, device=device)
    mD = torch.zeros((m, n), dtype=torch.float32, device=device)

    mX_idx = Pts[1].clone().detach().to(dtype=torch.int64, device=device)
    mY_idx = Pts[0].clone().detach().to(dtype=torch.int64, device=device)

    mX[mX_idx, mY_idx] = Pts[0] - torch.round(Pts[0])
    mY[mX_idx, mY_idx] = Pts[1] - torch.round(Pts[1])
    mD[mX_idx, mY_idx] = Pts[2]

    KmX = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmY = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)
    KmD = torch.zeros((ng, ng, m - ng, n - ng), dtype=torch.float32, device=device)

    # KmX = torch.zeros((ng, ng), dtype=torch.float32, device=device)
    # KmY = torch.zeros((ng, ng), dtype=torch.float32, device=device)
    # KmD = torch.zeros((ng, ng), dtype=torch.float32, device=device)

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + j
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]

    S = torch.zeros_like(KmD[0, 0], device=device)
    Y = torch.zeros_like(KmD[0, 0], device=device)

    for i in range(ng):
        for j in range(ng):
            # s = 1 / torch.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2)
            s = 1 / torch.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2 + epsilon)
            Y += s * KmD[i, j]
            S += s

    S[S == 0] = 1
    out = torch.zeros((m, n), dtype=torch.float32, device=device)
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    # return out.cpu()  # 최종 결과를 CPU로 이동
    return out # 최종 결과를 GPU

def colormap(disp):
    """"Color mapping for disp -- [H, W] -> [3, H, W]"""
    disp_np = disp.cpu().numpy()        # tensor -> numpy
    # disp_np = disp
    # vmax = np.percentile(disp_np, 95)
    vmin = disp_np.min()
    vmax = disp_np.max()
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')  #magma, plasma, etc.
    colormapped_im = (mapper.to_rgba(disp_np)[:, :, :3])
    # return colormapped_im.transpose(2, 0, 1)
    # colormapped_tensor = torch.from_numpy(colormapped_im).permute(2, 0, 1).to(dtype=torch.float32)
    colormapped_tensor = torch.from_numpy(colormapped_im)
    return colormapped_tensor

def trim_corrs(points ,num_kp=30000):
    length = points.shape[0]
#         print ("number of keypoint before trim : {}".format(length))
    if length >= num_kp:
        # mask = np.random.choice(length, num_kp)
        mask = torch.randperm(length)[:num_kp]
        return points[mask]
    else:
        # mask = np.random.choice(length, num_kp - length)
        mask = torch.randint(0, length, (num_kp - length,))
        # return np.concatenate([points, points[mask]], axis=0)
        return torch.cat([points, points[mask]], dim=0)