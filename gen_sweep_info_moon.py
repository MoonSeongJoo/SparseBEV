import os
import mmcv
import tqdm
import pickle
import argparse
import numpy as np
from nuscenes import NuScenes
from pyquaternion import Quaternion

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', default='data/nuscenes')
parser.add_argument('--version', default='v1.0-trainval')
args = parser.parse_args()

def get_cam_info(nusc, sample_data):
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
    
    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix
    cam_intrinsic = np.array(cs_record['camera_intrinsic'])

    sensor2global_rotation = sensor2ego_rotation.T @ ego2global_rotation.T
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    return {
        'data_path': os.path.join(args.data_root, sample_data['filename']),
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'cam_intrinsic': cam_intrinsic,
        'timestamp': sample_data['timestamp'],
    }

def get_lidar_info(nusc, sample_data):
    pose_record = nusc.get('ego_pose', sample_data['ego_pose_token'])
    cs_record = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])

    sensor2ego_translation = cs_record['translation']
    ego2global_translation = pose_record['translation']
    sensor2ego_rotation = Quaternion(cs_record['rotation']).rotation_matrix
    ego2global_rotation = Quaternion(pose_record['rotation']).rotation_matrix

    sensor2global_rotation = sensor2ego_rotation.T @ ego2global_rotation.T
    sensor2global_translation = sensor2ego_translation @ ego2global_rotation.T + ego2global_translation

    return {
        'data_path': os.path.join(args.data_root, sample_data['filename']),
        'sensor2global_rotation': sensor2global_rotation,
        'sensor2global_translation': sensor2global_translation,
        'timestamp': sample_data['timestamp'],
    }

def add_sweep_info(nusc, sample_infos):
    for curr_id in tqdm.tqdm(range(len(sample_infos['infos']))):
        sample = nusc.get('sample', sample_infos['infos'][curr_id]['token'])
    # for curr_id in tqdm.tqdm(range(len(sample_infos['data_list']))):
    #     sample = nusc.get('sample', sample_infos['data_list'][curr_id]['token'])

        cam_types = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
        ]

        lidar_types = [
            'LIDAR_TOP'
        ]

        curr_cams = dict()
        for cam in cam_types:
            curr_cams[cam] = nusc.get('sample_data', sample['data'][cam])

        curr_lidars = dict()
        for lidar in lidar_types:
            curr_lidars[lidar] = nusc.get('sample_data', sample['data'][lidar])

        for cam in cam_types:
            sample_data = nusc.get('sample_data', sample['data'][cam])
            sweep_cam = get_cam_info(nusc, sample_data)
            sample_infos['infos'][curr_id]['cams'][cam].update(sweep_cam)
            # sample_infos['infos'][curr_id]['images'][cam].update(sweep_cam)
        
        if 'lidars' not in sample_infos['infos'][curr_id]:
            sample_infos['infos'][curr_id]['lidars'] = {}

        for lidar in lidar_types:
            sample_data = nusc.get('sample_data', sample['data'][lidar])
            sweep_lidar = get_lidar_info(nusc, sample_data)
            sample_infos['infos'][curr_id]['lidars'][lidar] = sweep_lidar
            # sample_infos['infos'][curr_id]['lidar_points'][lidar] = sweep_lidar

        # # remove unnecessary
        # for cam in cam_types:
        #     del sample_infos['infos'][curr_id]['cams'][cam]['sample_data_token']
        #     del sample_infos['infos'][curr_id]['cams'][cam]['sensor2ego_translation']
        #     del sample_infos['infos'][curr_id]['cams'][cam]['sensor2ego_rotation']
        #     del sample_infos['infos'][curr_id]['cams'][cam]['ego2global_translation']
        #     del sample_infos['infos'][curr_id]['cams'][cam]['ego2global_rotation']

        # for lidar in lidar_types:
        #     del sample_infos['infos'][curr_id]['lidars'][lidar]['sample_data_token']
        #     del sample_infos['infos'][curr_id]['lidars'][lidar]['sensor2ego_translation']
        #     del sample_infos['infos'][curr_id]['lidars'][lidar]['sensor2ego_rotation']
        #     del sample_infos['infos'][curr_id]['lidars'][lidar]['ego2global_translation']
        #     del sample_infos['infos'][curr_id]['lidars'][lidar]['ego2global_rotation']

        sweep_infos = []
        if sample['prev'] != '':  # add sweep frame between two key frame
            for _ in range(5):
                sweep_info = dict()
                for cam in cam_types: 
                    if curr_cams[cam]['prev'] == '':    
                        sweep_info = sweep_infos[-1] 
                        break
                    sample_data = nusc.get('sample_data', curr_cams[cam]['prev'])
                    sweep_cam = get_cam_info(nusc, sample_data)
                    curr_cams[cam] = sample_data
                    sweep_info[cam] = sweep_cam
                
                for lidar in lidar_types:
                    if curr_lidars[lidar]['prev'] == '':
                        sweep_info = sweep_infos[-1]
                        break
                    sample_data = nusc.get('sample_data', curr_lidars[lidar]['prev'])
                    sweep_lidar = get_lidar_info(nusc, sample_data)
                    curr_lidars[lidar] = sample_data
                    sweep_info[lidar] = sweep_lidar

                sweep_infos.append(sweep_info)

        sample_infos['infos'][curr_id]['sweeps'] = sweep_infos

    return sample_infos

if __name__ == '__main__':
    nusc = NuScenes(args.version, args.data_root)

    if args.version == 'v1.0-trainval':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_train_sweep.pkl'), 'rb'))
        # sample_infos = transfer_data_list_to_infos(sample_infos,args.version)
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_train_sweep_moon.pkl'))

        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_val_sweep.pkl'), 'rb'))
        # sample_infos = transfer_data_list_to_infos(sample_infos,args.version)
        sample_infos = add_sweep_info(nusc, sample_infos)
        mmcv.dump(sample_infos, os.path.join(args.data_root, 'nuscenes_infos_val_sweep_moon.pkl'))

    elif args.version == 'v1.0-test':
        sample_infos = pickle.load(open(os.path.join(args.data_root, 'nuscenes_infos_test_sweep.pkl'), 'rb'))
        # sample_infos = transfer_data_list_to_infos(sample_infos,args.version)
        sample_infos = add_sweep_info(nusc, sample_infos)
