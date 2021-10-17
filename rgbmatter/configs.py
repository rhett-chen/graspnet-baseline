import os


def get_config_rgbmatter():
    '''
     - return the config dict
    '''
    config = dict()
    config['heatmap_width'] = 1280
    config['heatmap_height'] = 720
    config['scene_id'] = 22
    config['index_str'] = '0100'
    config['camera'] = 'kinect'
    config['collision_detection_choice'] = 'point_cloud'  # point_cloud or full_model

    config['dataset_path'] = "/media/bot/980A6F5E0A6F3880/datasets/graspnet/"
    config['heatmap_path'] = '/home/bot/grasp/graspnet-baseline/rgbmatter/heatmap_6Dpose/heatmap'
    config['res_6D_pose_path'] = '/home/bot/grasp/graspnet-baseline/rgbmatter/heatmap_6Dpose/6Dpose'
    config['grasp_gt_path'] = '/home/bot/grasp/graspGT'

    config['scene_id_str'] = 'scene_' + str(config['scene_id']).zfill(4)
    config['scene_heatmap_path'] = os.path.join(config['heatmap_path'], config['scene_id_str'], config['camera'])
    config['scene_res_6Dpose_path'] = os.path.join(config['res_6D_pose_path'], config['scene_id_str'], config['camera'])

    return config
