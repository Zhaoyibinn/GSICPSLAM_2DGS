import open3d as o3d
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def rodrigues_rotation_formula(v, theta):
    v = np.array(v, dtype=float)
    v_norm = v / np.linalg.norm(v)
    K = np.array([[0, -v_norm[2], v_norm[1]], [v_norm[2], 0, -v_norm[0]], [-v_norm[1], v_norm[0], 0]], dtype=float)
    K2 = np.dot(K, K)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = np.eye(3) + sin_theta * K + (1 - cos_theta) * K2
    return R

ply_path = "/media/zhaoyibin/common/3DRE/3DGS/GS_ICP_SLAM_ORIGIN/GS_ICP_SLAM/zyb_tools/2dgs_dtu_7000.ply"
pcd = o3d.io.read_point_cloud(ply_path)
plydata = PlyData.read(ply_path)
qs = []
for name in ["rot_0","rot_1","rot_2","rot_3"]:
    qs.append(np.array(plydata['vertex'][name]))
qs = np.array(qs).T
rot = R.from_quat(qs)





rotation_matrix = rot.as_matrix()

# maybe
normals = rot.as_matrix()[:,:,2]
pcd.normals = o3d.utility.Vector3dVector(normals)

print(pcd.points)
nb_neighbors = 200
std_ratio = 1.0
print(pcd.points)
cl, ind = pcd.remove_statistical_outlier(
    nb_neighbors=nb_neighbors,
    std_ratio=std_ratio
)
pcd = pcd.select_by_index(ind)
print(pcd.points)
#
# voxel_size = 0.5  # 体素大小为0.1x0.1x0.1
#
# # 执行体素下采样
# pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
# print(pcd.points)

# normals = np.zeros_like(pcd.points)  # 创建一个与点云形状相同的全零数组
# normals[:, 2] = 1.0

# pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

voxel_size = 0.03  # 体素大小为0.1x0.1x0.1
pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
print(pcd.points)

o3d.visualization.draw_geometries([pcd], point_show_normal=True)
# o3d.visualization.draw_geometries([pcd])