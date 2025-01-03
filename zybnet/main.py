import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import imp
import os

def data_loader_trans(img):
    img = ((img/255.)*2-1).astype(np.float32)
    # 这里用了参考视图的归一化的
    return img


img_path = "/media/zhaoyibin/common/3DRE/3DGS/GS_ICP_SLAM_ORIGIN/GS_ICP_SLAM_2d/dataset/TUM/rgbd_dataset_freiburg2_xyz/rgb/1311867170.694205.png"
depth_path ="/media/zhaoyibin/common/3DRE/3DGS/GS_ICP_SLAM_ORIGIN/GS_ICP_SLAM_2d/dataset/TUM/rgbd_dataset_freiburg2_xyz/depth/1311867170.680532.png"

rgb = cv2.imread(img_path)
depth = cv2.imread(depth_path)
rgb = cv2.resize(rgb, (640, 960), interpolation=cv2.INTER_LINEAR)
depth = cv2.resize(depth, (640, 960), interpolation=cv2.INTER_LINEAR)
rgb = data_loader_trans(rgb)
rgb_batch = torch.from_numpy(rgb).unsqueeze(0).unsqueeze(0).view(1, 1, 3, 640, -1)


module = 'zybnet.network'
path = 'zybnet/network.py'
batch = {}
batch.update({'rgb': rgb_batch})
network = imp.load_source(module, path).Network()

load_params = "zybnet/weights/dtu_pretrained/latest.pth"

if os.path.exists(load_params):
    pretext_model = torch.load(load_params)["net"]
    state_dict = {k: v for k, v in pretext_model.items() if k.startswith("feature_net")}
    network.load_state_dict(state_dict)
feature = network(batch)


print("end")