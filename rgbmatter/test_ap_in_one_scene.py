import numpy as np
import os
from graspnetAPI import GraspGroup
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
from eval_all_without_multiprocess import GraspNetEval


if __name__ == '__main__':
    scene = 0
    grasp_gt_path = '/home/bot/grasp/graspGT'
    res_6D_path = '/home/bot/grasp/graspnet-baseline/rgbmatter/heatmap_6Dpose/6Dpose'

    dataset_path = '/media/bot/980A6F5E0A6F3880/datasets/graspnet/'
    ge = GraspNetEval(dataset_path, camera='kinect', split='test')
    ge.eval_scene(scene_id=scene, dump_folder=grasp_gt_path)
