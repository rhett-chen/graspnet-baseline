__author__ = 'mhgou'
__version__ = '1.0'

from graspnetAPI import GraspNet
from graspnetAPI import GraspGroup
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from rgbmatter.configs import get_config_rgbmatter
from rgbmatter.util import collision_detection_with_full_models


config = get_config_rgbmatter()
graspnet_root = config['dataset_path']  # ROOT PATH FOR GRASPNET

sceneId = 22
annId = 100

# initialize a GraspNet instance
g = GraspNet(graspnet_root, camera='kinect', split='train')

# load grasps of scene 1 with annotation id = 3, camera = kinect and fric_coef_thresh = 0.2
grasp = g.loadGrasp(sceneId=sceneId, annId=annId, format='6d', camera='kinect', fric_coef_thresh=0.1)
print('grasp num after load grasp: ', grasp.grasp_group_array.shape)
grasp = collision_detection_with_full_models(grasp, annId)
print('grasp after collision detection: ', grasp.grasp_group_array.shape)
np.save('a.npy', grasp.grasp_group_array)
