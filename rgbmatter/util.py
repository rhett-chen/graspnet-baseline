import numpy as np
import torch
import random
import os
import pickle
from graspnetAPI.utils.config import get_config
from graspnetAPI.utils.xmlhandler import xmlReader
import open3d as o3d
from graspnetAPI.utils.eval_utils import create_table_points, parse_posevector, transform_points, \
     load_dexnet_model, voxel_sample_points, collision_detection, compute_closest_points
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from rgbmatter.configs import get_config_rgbmatter


def create_point_cloud_from_depth_image(depth, camera, organized=True):
    """ Generate point cloud using depth image only.

        Input:
            depth: [numpy.ndarray, (H,W), numpy.float32]
                depth image
            camera: [CameraInfo]
                camera intrinsics
            organized: bool
                whether to keep the cloud in image shape (H,W,3)

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    cloud = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        cloud = cloud.reshape([-1, 3])
    return cloud


def create_xymaps_from_point_cloud(cloud, cam_int_mat):
    """ Generate point cloud using depth image only.
        Input:
            cloud: [numpy.ndarray, (H,W,3), numpy.float32]
                depth image

        Output:
            cloud: [numpy.ndarray, (H,W,3)/(H*W,3), numpy.float32]
                generated cloud, (H,W,3) for organized=True, (H*W,3) for organized=False
    """
    points_x, points_y, points_z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    cam_cx, cam_cy = cam_int_mat[0, 2], cam_int_mat[1, 2]
    cam_fx, cam_fy = cam_int_mat[0, 0], cam_int_mat[1, 1]

    xmap = points_x * cam_fx / points_z + cam_cx
    ymap = points_y * cam_fy / points_z + cam_cy
    print('x y map maximum: ', max(xmap), max(ymap))
    xymaps = np.stack([xmap, ymap], axis=-1)

    return xymaps


def generate_grasp_views_half(N=300, phi=(np.sqrt(5)-1)/2, center=np.zeros(3), r=1):
    """ View sampling on a unit sphere using Fibonacci lattices.
        Ref: https://arxiv.org/abs/0912.4540

        Input:
            N: [int]
                number of sampled views
            phi: [float]
                constant for view coordinate calculation, different phi's bring different distributions, default: (sqrt(5)-1)/2
            center: [np.ndarray, (3,), np.float32]
                sphere center
            r: [float]
                sphere radius

        Output:
            views: [torch.FloatTensor, (N,3)]
                sampled view coordinates
    """
    views = []
    for i in range(int(N/2), N):
        zi = (2 * i + 1) / N - 1
        xi = np.sqrt(1 - zi**2) * np.cos(2 * i * np.pi * phi)
        yi = np.sqrt(1 - zi**2) * np.sin(2 * i * np.pi * phi)
        views.append([xi, yi, zi])
    views = r * np.array(views) + center
    return views.astype(np.float32)
    # return torch.from_numpy(views.astype(np.float32))


def heatmap_to_xyz_ori(heatmap, point_cloud, grasp_gt):
    """
    :param heatmap: 72 * 96
    :param point_cloud: 720 * 1280 *3
    :return: xyz vx vy vz angle
    """
    print('heatmap shape: ', heatmap.shape)
    height = heatmap.shape[1]
    width = heatmap.shape[2]
    grasp_views = generate_grasp_views_half(N=120)

    heatmap = heatmap.reshape((60, 6, heatmap.shape[1], heatmap.shape[2]))  # 360 * hegiht * width
    incomplete_pose = []
    threshold = 0.001
    for vi in range(60):
        for ai in range(6):
            xys = np.where(heatmap[vi, ai] > 0)  # tuple (array[all x], array[all y])
            for index in range(xys[0].size):
                if random.random() < 0.5:
                    continue
                xy = [xys[0][index], xys[1][index]]  # in heatmap 72*96 resolution
                xyz = list(point_cloud[int(xy[0] * 720 / height), int(xy[1] * 1280 / width)])
                xyz.extend(grasp_views[vi])
                xyz.append(ai * np.pi / 6)
                incomplete_pose.append(xyz)

                # search in ground truth grasps
                # gt_index = np.where(np.logical_and(abs(grasp_gt[:, 13] - xyz[0]) <= threshold,
                #                                    abs(grasp_gt[:, 14] - xyz[1]) <= threshold))
                # if gt_index[0].size < 1:
                #     continue
                # else:
                #     xyz = list(grasp_gt[gt_index[0][0]][13:16])
                #
                #    # xyz.extend(grasp_views[vi])
                #     vx, vy, vz = grasp_gt[gt_index[0][0]][4], grasp_gt[gt_index[0][0]][7], grasp_gt[gt_index[0][0]][10]
                #     xyz.extend([vx, vy, vz])  # use gt approaching vec
                #
                #     # xyz.append(ai * np.pi / 6)
                #     angle = np.arccos(np.clip(grasp_gt[gt_index[0][0]][12] / np.sqrt(vx * vx + vy * vy), -1, 1))
                #     xyz.append(angle)  # use gt angle
                #
                #     xyz.append(grasp_gt[gt_index[0][0]][1])  # gt width
                #     xyz.append(grasp_gt[gt_index[0][0]][3])  # gt depth
                #
                #     incomplete_pose.append(xyz)
    return np.array(incomplete_pose)


def collision_detection_with_full_models(gg, index):
    my_config = get_config_rgbmatter()
    camera_pose = np.load(os.path.join(my_config['dataset_path'], 'scenes', my_config['scene_id_str'],
                                       my_config['camera'], 'camera_poses.npy'))[index]
    align_mat = np.load(os.path.join(my_config['dataset_path'], 'scenes', my_config['scene_id_str'],
                                     my_config['camera'], 'cam0_wrt_table.npy'))
    scene_reader = xmlReader(os.path.join(my_config['dataset_path'], 'scenes', my_config['scene_id_str'],
                                          my_config['camera'], 'annotations', str(index).zfill(4) + '.xml'))
    model_dir = os.path.join(my_config['dataset_path'], 'models')

    posevectors = scene_reader.getposevectorlist()
    obj_list = []
    pose_list = []
    for posevector in posevectors:
        obj_idx, mat = parse_posevector(posevector)
        obj_list.append(obj_idx)
        pose_list.append(mat)
    table = create_table_points(1.0, 1.0, 0.05, dx=-0.5, dy=-0.5, dz=-0.05, grid_size=0.008)
    table_trans = transform_points(table, np.linalg.inv(np.matmul(align_mat, camera_pose)))
    obj_list = []
    model_list = []
    dexmodel_list = []
    for posevector in posevectors:
        obj_idx, _ = parse_posevector(posevector)
        obj_list.append(obj_idx)
    for obj_idx in obj_list:
        model = o3d.io.read_point_cloud(os.path.join(model_dir, '%03d' % obj_idx, 'nontextured.ply'))
        dex_cache_path = os.path.join(my_config['dataset_path'], "dex_models", '%03d.pkl' % obj_idx)
        if os.path.exists(dex_cache_path):
            with open(dex_cache_path, 'rb') as f:
                dexmodel = pickle.load(f)
        else:
            dexmodel = load_dexnet_model(os.path.join(model_dir, '%03d' % obj_idx, 'textured'))
        points = np.array(model.points)
        model_list.append(points)
        dexmodel_list.append(dexmodel)

    models = list()
    for model in model_list:
        model_sampled = voxel_sample_points(model, 0.008)
        models.append(model_sampled)

    num_models = len(models)
    # assign grasps to object
    # merge and sample scene
    model_trans_list = list()
    seg_mask = list()
    for i, model in enumerate(models):
        model_trans = transform_points(model, pose_list[i])
        seg = i * np.ones(model_trans.shape[0], dtype=np.int32)
        model_trans_list.append(model_trans)
        seg_mask.append(seg)
    seg_mask = np.concatenate(seg_mask, axis=0)
    scene = np.concatenate(model_trans_list, axis=0)

    # assign grasps
    indices = compute_closest_points(gg.translations, scene)
    model_to_grasp = seg_mask[indices]
    grasp_list = list()
    for i in range(num_models):
        grasp_i = gg[model_to_grasp == i]
        grasp_list.append(grasp_i.grasp_group_array)

    scene = np.concatenate([scene, table_trans])
    # collision detection
    collision_mask_list, empty_list = collision_detection(
        grasp_list, model_trans_list, dexmodel_list, pose_list, scene, outlier=0.05, empty_thresh=10, return_dexgrasps=False)

    grasp_list, collision_mask_list = np.concatenate(grasp_list), np.concatenate(collision_mask_list)
    grasp_list = grasp_list[np.logical_not(collision_mask_list)]
    gg.grasp_group_array = grasp_list
    return gg
