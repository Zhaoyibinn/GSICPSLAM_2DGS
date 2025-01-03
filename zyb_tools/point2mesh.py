import open3d as o3d
import numpy as np

ply_path = "/media/zhaoyibin/common/3DRE/3DGS/GS_ICP_SLAM_ORIGIN/GS_ICP_SLAM_2d/output/room0/scene.ply"
ply_path = "/media/zhaoyibin/common/3DRE/3DGS/GS_ICP_SLAM_ORIGIN/GS_ICP_SLAM/output/room0/scene.ply"
pcd = o3d.io.read_point_cloud(ply_path)
pcd.estimate_normals()
radius = 3 * np.mean(pcd.compute_nearest_neighbor_distance())
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector([radius, radius * 2]))
o3d.visualization.draw_geometries([mesh])