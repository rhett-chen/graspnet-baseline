import numpy as np
import open3d as o3d
from PIL import Image
import os
import scipy.io as scio
from graspnetAPI import GraspGroup
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from utils.data_utils import CameraInfo, create_point_cloud_from_depth_image


index = '0000'
scene_id = 10
scene_id_str = 'scene_' + str(scene_id).zfill(4)

abs_path = 'F:\\research\\Liu\\relevant_projects\\graspnet-baseline\\'
grasp = np.load(abs_path + 'heatmap_6Dpose/' + scene_id_str + '/res_6d/' + index + '.npy')
# grasp = np.load(abs_path + 'heatmap_6Dpose/' + scene_id_str + '/gt/' + index + '.npy')

gg = GraspGroup(grasp)
print('grasp shape: ', gg.grasp_group_array.shape)
color = np.array(Image.open(abs_path + "heatmap_6Dpose/" + scene_id_str + "/rgb/" + index + '.png'), dtype=np.float32) / 255.0
color = color.reshape((-1, 3))
depth = np.array(Image.open(abs_path + "heatmap_6Dpose/" + scene_id_str + "/depth/" + index + '.png'))
meta = scio.loadmat(abs_path + 'heatmap_6Dpose/' + scene_id_str + '/meta/' + index + '.mat')
intrinsic = meta['intrinsic_matrix']
factor_depth = meta['factor_depth']
camera = CameraInfo(1280.0, 720.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
point_cloud = create_point_cloud_from_depth_image(depth, camera, organized=False)  # 720 * 1280 * 3 numpy.ndarray

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(point_cloud.astype(np.float32))
cloud.colors = o3d.utility.Vector3dVector(color.astype(np.float32))

gg = gg.nms(0.03, 30.0/180*np.pi)
gg = gg.random_sample(70)
grippers = gg.to_open3d_geometry_list()
o3d.visualization.draw_geometries([cloud, *grippers])
