import numpy as np
import os
from PIL import Image
import torch
import sys
import scipy.io as scio
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from knn.knn_modules import knn
from rgbmatter.util import generate_grasp_views_half, create_xymaps_from_point_cloud
from rgbmatter.configs import get_config_rgbmatter


if __name__ == '__main__':
    config = get_config_rgbmatter()

    if not os.path.exists(config['scene_heatmap_path']):
        os.makedirs(config['scene_heatmap_path'])

    for i in range(3, 256):
        print('processing scene {}, image {}'.format(config['scene_id'], i))
        # grasp label in graspnetAPI format [N*17]
        # grasp_gt = np.load(os.path.join(config['grasp_gt_path'], config['scene_id_str'], config['camera'],
        #                                 str(i).zfill(4) + '.npy'))
        grasp_gt = np.load('a.npy')
        # for this format, rotation matrix's first column is approaching vector because gripper's orientation is x-axis
        up_inds = grasp_gt[:, 10] > 0
        grasp_gt = grasp_gt[up_inds]

        xyz = grasp_gt[:, 13:16]  # m
        # x,y from camera coordinate to image coordinate, use camera intrinsic matrix
        meta = scio.loadmat(os.path.join(config['dataset_path'], 'scenes', config['scene_id_str'], config['camera'],
                                         'meta', str(i).zfill(4) + '.mat'))
        intrinsic = meta['intrinsic_matrix']
        xy_map = create_xymaps_from_point_cloud(xyz, intrinsic)  # x, y in image's original resolution
        y_index = xy_map[:, 1] < 720
        x_index = xy_map[:, 0] < 1280
        x_y_index = np.logical_and(y_index, x_index)
        xy_map = xy_map[x_y_index]
        grasp_gt = grasp_gt[x_y_index]
        x, y = xy_map[:, 0], xy_map[:, 1]
        vx, vy, vz = grasp_gt[:, 4], grasp_gt[:, 7], grasp_gt[:, 10]
        print('grasp num we use in gt: ', xy_map.shape[0])
        # in-plane rotation angle's calculation refer to batch_viewpoint_params_to_matrix function in loss_util.py
        # cos(in-plane rotation angle) may be = 1.00000002, clip it to [-1, 1]
        gt_angle = np.arccos(np.clip(grasp_gt[:, 12] / np.sqrt(vx * vx + vy * vy), -1, 1))
        gt_views = torch.tensor(np.stack([vx, vy, vz], -1))
        gt_views = gt_views.transpose(0, 1).contiguous().unsqueeze(0)

        grasp_views = torch.from_numpy(generate_grasp_views_half(N=120))  # sampling N views in sphere space for *AVN*
        grasp_views_trans_ = grasp_views.transpose(0, 1).contiguous().unsqueeze(0)  # from N*3 to 1*3*N

        # for each GT find the closest view/angle/position in the template, 6D orientation match with 2D view and angle
        # x and y are in image's original resolution 1280*720, AVN input shape is 384*288, label heatmap shape is 96*72
        view_inds = knn(grasp_views_trans_, gt_views, k=1).squeeze() - 1  # (N,1) , knn (ref, query),
        angle_inds = torch.tensor(np.floor((gt_angle - 1e-5) / np.pi * 6.0).astype(int))  # (N, 1)
        x, y = torch.tensor(x), torch.tensor(y)
        x = torch.clamp(torch.round(x / (1280. / config['heatmap_width'])), 0,
                        config['heatmap_width'] - 1).to(torch.long)
        y = torch.clamp(torch.round(y / (720. / config['heatmap_height'])), 0,
                        config['heatmap_height'] - 1).to(torch.long)

        heatmap = torch.zeros(60, 6, config['heatmap_height'], config['heatmap_width'])
        heatmap[view_inds, angle_inds, y, x] = 1  # S
        heatmap = heatmap.view(60 * 6, config['heatmap_height'], config['heatmap_width'])
        print('grasp num in heatmap: ', torch.sum(heatmap > 0))
        np.save(os.path.join(config['scene_heatmap_path'], str(i).zfill(4) + '.npy'), heatmap.numpy())
        break