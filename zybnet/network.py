from .featurenet import FeatureNet
import torch
import torch.nn as nn
import os

class Network(nn.Module):
    def __init__(self,):
        super(Network, self).__init__()
        self.feature_net = FeatureNet()


    def forward_feat(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B*S, C, H, W)
        feat2, feat1, feat0 = self.feature_net(x)
        feats = {
                'level_2': feat0.reshape((B, S, feat0.shape[1], H, W)),
                'level_1': feat1.reshape((B, S, feat1.shape[1], H//2, W//2)),
                'level_0': feat2.reshape((B, S, feat2.shape[1], H//4, W//4)),
                }
        return feats

    def forward(self, batch):
        B, _, _, H_img, W_img = batch['rgb'].shape
        ret = {}
        feats = self.forward_feat(batch['rgb'])
        ret.update({'feats': feats})
        # 返回了三个等级的特征，124降采样，特征深度也不断翻倍
        return ret
