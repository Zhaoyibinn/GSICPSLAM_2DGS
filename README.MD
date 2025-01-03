forked from GS ICP SLAM and 2DGS

# 2RGBD GS-ICP SLAM

2DGS + GS-ICP SLAM

主要的修改就是把原本的3DGS替换为了2DGS，并且设置2DGS的rot不更新

环境就是在GS ICP SLAM官方的基础上，把光栅化部分换成2DGS的

python gs_icp_slam.py