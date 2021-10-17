import numpy as np
import os
from PIL import Image
import torch
import scipy.io as scio
import sys
from graspnetAPI import GraspGroup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils.loss_utils import batch_viewpoint_params_to_matrix
from utils.data_utils import CameraInfo
from utils.collision_detector import ModelFreeCollisionDetector
from rgbmatter.util import create_point_cloud_from_depth_image, heatmap_to_xyz_ori, collision_detection_with_full_models
from rgbmatter.configs import get_config_rgbmatter


if __name__ == '__main__':
    config = get_config_rgbmatter()

    if not os.path.exists(config['scene_res_6Dpose_path']):
        os.makedirs(config['scene_res_6Dpose_path'])

    for i in range(3, 256):   # 256 images
        print('processing scene {}, image {}\n'.format(config['scene_id'], i))
        grasp_gt = np.load(os.path.join(config['grasp_gt_path'], config['scene_id_str'], config['camera'],
                                        str(i).zfill(4) + '.npy'))
        # for this format, rotation matrix's first column is approaching vector because gripper's orientation is x-axis
        up_inds = grasp_gt[:, 10] > 0
        grasp_gt = grasp_gt[up_inds]

        heatmap = np.load(os.path.join(config['scene_heatmap_path'], str(i).zfill(4) + '.npy'))
        depth = np.array(Image.open(os.path.join(config['dataset_path'], 'scenes', config['scene_id_str'],
                                                 config['camera'], 'depth', str(i).zfill(4) + '.png')))   # mm
        meta = scio.loadmat(os.path.join(config['dataset_path'], 'scenes', config['scene_id_str'], config['camera'],
                                         'meta', str(i).zfill(4) + '.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']

        # generate cloud
        camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2],
                            factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)  # 720 * 1280 * 3 numpy.ndarray, m
        incomplete_pose = heatmap_to_xyz_ori(heatmap, cloud, grasp_gt)
        xyz = incomplete_pose[:, :3]
        print('grasp num before depth and width sampling: ', xyz.shape[0])

        approaching_vector = incomplete_pose[:, 3:6]
        angle = incomplete_pose[:, 6]

        # gt_width = incomplete_pose[:, 7]  # use gt width and depth
        # gt_depth = incomplete_pose[:, 8]
        # grasp_width = torch.from_numpy(gt_width).unsqueeze(-1)
        # grasp_depth = torch.from_numpy(gt_depth).unsqueeze(-1)
        grasp_width = torch.arange(3, 11) * 0.01  # sample grasp width , 10
        grasp_dist = torch.arange(-2, 3) * 0.01
        # grasp_depth = torch.arange(-2, 3) * 0.01  # sample grasp depth, 5

        grasp_num = incomplete_pose.shape[0]  # grasp num in heatmap
        grasp_score = torch.ones((grasp_num, 1))
        ori = batch_viewpoint_params_to_matrix(torch.from_numpy(approaching_vector), torch.from_numpy(angle))
        ori = ori.view(grasp_num, 9)
        xyz = torch.from_numpy(xyz)

        grasp_width = grasp_width.unsqueeze(-1).repeat(grasp_num, 5).view(-1, 1)  # 50 combinations of width and depth
        # grasp_depth = grasp_depth.unsqueeze(-1).repeat(grasp_num * 8, 1).view(-1, 1)
        grasp_dist = grasp_dist.unsqueeze(-1).repeat(grasp_num * 8, 1).view(-1, 1)
        ori = ori.repeat(1, 40).view(-1, 9)  # each one has 50 combinations of width and depth
        xyz = xyz.repeat(1, 40).view(-1, 3)  # each one 50 combination of width and depth
        # grasp_dist = torch.cat([grasp_dist, torch.zeros_like(grasp_dist), torch.zeros_like(grasp_dist)],
        #                        axis=-1).unsqueeze(-1).double()
        # grasp_dist = torch.bmm(ori.view(-1, 3, 3).double(), grasp_dist)
        # xyz = xyz + grasp_dist.squeeze()
        xyz[:, 2] = xyz[:, 2] + grasp_dist.squeeze()
        grasp_score = grasp_score.repeat(1, 40).view(-1, 1)  # each one 50 combination of width and depth

        obj_ids = -1 * torch.ones_like(grasp_score)
        grasp_height = 0.02 * torch.ones_like(grasp_score)
        grasp_depth = 0.03 * torch.ones_like(grasp_score)
        grasp_decode = torch.cat([grasp_score, grasp_width, grasp_height, grasp_depth, ori, xyz, obj_ids],
                                 axis=-1).numpy()
        print('grasp after sampling: ', grasp_decode.shape)

        if config['collision_detection_choice'] == 'point_cloud':
            gg = GraspGroup(grasp_decode)
            mfcdetector = ModelFreeCollisionDetector(cloud.reshape([-1, 3]),  # collision detection use point cloud
                                                     voxel_size=0.08)  # filter the collision and empty grasps
            collision_mask = mfcdetector.detect(gg, approach_dist=0.03, collision_thresh=0.0, empty_thresh=0.3,
                                                return_empty_grasp=True)
            final_gg = gg[np.logical_and(~collision_mask[0], ~collision_mask[1])]
        else:
            final_gg = GraspGroup(np.empty([0, 17]))
            for j in range(int(np.ceil(grasp_decode.shape[0] / 6000))):
                start = j * 6000
                print('processing grasps {} ~ {}'.format(start, start + 6000))
                gg = GraspGroup(grasp_decode[start:start + 6000])
                gg = collision_detection_with_full_models(gg, i)  # collision detection use full object models
                final_gg.grasp_group_array = np.concatenate([final_gg.grasp_group_array, gg.grasp_group_array])

        print('grasp shape after filter: ', final_gg.grasp_group_array.shape)  # (39999, 17)  why
        save_path = os.path.join(config['scene_res_6Dpose_path'], str(i).zfill(4) + '.npy')
        final_gg.save_npy(save_path)
        print('save {} successfully!!!'.format(save_path))
        break