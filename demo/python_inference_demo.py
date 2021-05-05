# should we align attribute predictions with corresponding predictions

# currently: treating them as separate tasks
# how do we want to integrate / align those attributes to the objects

# where do they generate the detections, that is where we want the attributes to go into
# inside get_bbox, they are looking at task of each text prediction and getting those results
# go to get_bboxes, search how to get attributes from detector

from mmdet3d.apis import init_detector, inference_detector, show_result_meshlab

import os
import pickle

config_file = 'mmdetection3d/configs/centerpoint/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = '../mmdetection3d/work_dirs/centerpoint_02pillar_second_secfpn_4x8_cyclic_20e_nus/epoch_1.pth'
checkpoint_file = 'mmdetection3d/work_dirs/centerpoint_02pillar_second_secfpn_dcn_4x8_cyclic_20e_nus/latest.pth'
# checkpoint_file = 'mmdetection3d/demo/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# test a single sample
pcd = 'mmdetection3d/demo/car.bin'
result, data = inference_detector(model, pcd)

centerpoint_result_path = os.path.join(os.getcwd(), 'centerpoint_results.pkl')
centerpoint_data_path = os.path.join(os.getcwd(), 'centerpoint_data.pkl')

with open(centerpoint_result_path, 'wb') as f:
    pickle.dump(result, f)

with open(centerpoint_data_path, 'wb') as f:
    pickle.dump(data, f)


print("Done")
# show the results
# out_dir = './'
# show_result_meshlab(data, result, out_dir)
