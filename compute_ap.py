import numpy as np
import os

ap_scenes_path = '/home/zibo/graspnet-baseline/logs/dump_rs/ap_scenes'
acc_all = []

for index in range(100, 190):
    path = os.path.join(ap_scenes_path, str(index).zfill(4) + '.npy')
    acc_c = np.load(path)
    acc_all.append(acc_c)

acc_all = np.array(acc_all)
# 90 scenes * 256 images * 50 top_k * 6 len(list_coe_of_friction = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
print('acc shape: ', acc_all.shape)
ap_all = np.mean(acc_all)
ap_seen_all = np.mean(acc_all[0:30])
ap_unseen_all = np.mean(acc_all[30:60])
ap_novel_all = np.mean(acc_all[60:90])
print('AP: ', ap_all)
print('AP Seen: ', ap_seen_all)
print('AP Unseen: ', ap_unseen_all)
print('AP Novel: ', ap_novel_all)

ap_all_2 = np.mean(acc_all[:, :, :, 0])
ap_all_4 = np.mean(acc_all[:, :, :, 1])
ap_all_8 = np.mean(acc_all[:, :, :, 3])

ap_seen_2 = np.mean(acc_all[0:30, :, :, 0])
ap_seen_4 = np.mean(acc_all[0:30, :, :, 1])
ap_seen_8 = np.mean(acc_all[0:30, :, :, 3])

ap_unseen_2 = np.mean(acc_all[30:60, :, :, 0])
ap_unseen_4 = np.mean(acc_all[30:60, :, :, 1])
ap_unseen_8 = np.mean(acc_all[30:60, :, :, 3])


ap_novel_2 = np.mean(acc_all[60:90, :, :, 0])
ap_novel_4 = np.mean(acc_all[60:90, :, :, 1])
ap_novel_8 = np.mean(acc_all[60:90, :, :, 3 ])

print('\nAP all 0.2: ', ap_all_2)
print('AP all 0.4: ', ap_all_4)
print('AP all 0.8: ', ap_all_8)

print('\nAP Seen 0.2: ', ap_seen_2)
print('AP Seen 0.4: ', ap_seen_4)
print('AP Seen 0.8: ', ap_seen_8)

print('\nAP Unseen 0.2: ', ap_unseen_2)
print('AP Unseen 0.4: ', ap_unseen_4)
print('AP Unseen 0.8: ', ap_unseen_8)

print('\nAP Novel 0.2: ', ap_novel_2)
print('AP Novel 0.4: ', ap_novel_4)
print('AP Novel 0.8: ', ap_novel_8)
