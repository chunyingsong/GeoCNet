import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from mmcv.cnn import ConvModule
from mmseg.ops import resize


def build_ms2one(config):
    config = copy.deepcopy(config)
    t = config.pop('type')
    if t == 'Naive':
        return Naive(**config)
    elif t == 'DualAttentionFusion':
        return DualAttentionFusion(**config)


class Naive(nn.Module):
    def __init__(self, inc, outc, kernel_size=1):
        super().__init__()
        self.layer = nn.Conv2d(inc, outc, kernel_size=1)

    def forward(self, ms_feats):
        out = self.layer(torch.cat([
            F.interpolate(tmp, ms_feats[0].shape[-2:],
                          mode='bilinear') for tmp in ms_feats], dim=1))
        return out
        
class DualAttentionFusion(nn.Module):
    def __init__(self, inc, outc, num_scales=4, 
                 use_spatial=True, use_channel=True):
        super().__init__()
        self.num_scales = num_scales
        if not isinstance(inc, (tuple, list)):
            inc = [inc for _ in range(num_scales)]
        self.inc = inc
        self.outc = outc
        
        self.transforms = nn.ModuleList([
            nn.Conv2d(inc[i], outc, kernel_size=1) 
            for i in range(num_scales)
        ])
        
        self.use_channel = use_channel
        if use_channel:
            self.channel_attentions = nn.ModuleList([
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(outc, outc // 4, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(outc // 4, outc, kernel_size=1),
                    nn.Sigmoid()
                ) for _ in range(num_scales)
            ])
        
        self.use_spatial = use_spatial
        if use_spatial:
            self.spatial_attentions = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=7, padding=3),
                    nn.Sigmoid()
                ) for _ in range(num_scales)
            ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(outc, outc, kernel_size=3, padding=1),
            nn.BatchNorm2d(outc),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        feats = []
        target_size = x[0].shape[2:]
        
        for i in range(self.num_scales):
            feat = self.transforms[i](x[i])
            
            if self.use_channel:
                channel_att = self.channel_attentions[i](feat)
                feat = feat * channel_att
            
            if self.use_spatial:
                avg_mask = torch.mean(feat, dim=1, keepdim=True)
                max_mask, _ = torch.max(feat, dim=1, keepdim=True)
                spatial_mask = torch.cat([avg_mask, max_mask], dim=1)
                spatial_att = self.spatial_attentions[i](spatial_mask)
                feat = feat * spatial_att
            
            if i > 0:
                feat = F.interpolate(feat, target_size, mode='bilinear', align_corners=True)
            
            feats.append(feat)
        
        fused_feat = sum(feats)
        output = self.fusion(fused_feat)
        
        return output
