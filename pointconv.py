"""
Classification Model
Author: Wenxuan Wu
Date: September 2019
"""
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from pointconv_util import PointConvDensitySetAbstraction,  PointDeconv

class Point_enc(nn.Module):
    def __init__(self, num_classes = 21):
        # scannet : 21 seg classes
        super(Point_enc, self).__init__()
        feature_dim = 0
        bw= 0.05
        self.sa1 = PointConvDensitySetAbstraction(npoint=1024, nsample=38, in_channel=feature_dim + 3, mlp=[32,32,64], bandwidth = bw, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=256, nsample=38, in_channel=64 + 3, mlp=[64,64,128], bandwidth = 2*bw, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=64, nsample=38, in_channel=128 + 3, mlp=[128,128,256], bandwidth = 4*bw, group_all=False)
        self.sa4 = PointConvDensitySetAbstraction(npoint=36, nsample=38, in_channel=256 + 3, mlp=[256,256,512], bandwidth = 8*bw, group_all=False)

    def forward(self, xyz, feat):
        B, _, _ = xyz.shape
        print('batch size:', B)
        print('input xyz:', xyz.shape)
        print('input feat:', feat.shape)
        l1_xyz, l1_points = self.sa1(xyz, feat)
        print()
        print('setabstraction 1 (sa 1) output:')
        print('l1_xyz:', l1_xyz.shape)
        print('l1_points:', l1_points.shape)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        print()
        print('setabstraction 2 (sa 2) output:')
        print('l2_xyz:', l2_xyz.shape)
        print('l2_points:', l2_points.shape)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        print()
        print('setabstraction 3 (sa 3) output:')
        print('l3_xyz:', l3_xyz.shape)
        print('l3_points:', l3_points.shape)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        print()
        print('setabstraction 4 (sa 3) output:')
        print('l3_xyz:', l4_xyz.shape)
        print('l3_points:', l4_points.shape)
        # x = l3_points.view(B, 1024)
        # print("x shape (l3_points reshaped):", x.shape)
        # x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        # x = self.drop2(F.relu(self.bn2(self.fc2(x)))) 
        # x = self.fc3(x)
        # print("x shape (fc3 output):", x.shape)
        # x = F.log_softmax(x, -1)
        # print("x shape (log_softmax output):", x.shape)
        # return x, l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points
        return l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points
    
class Point_dec(nn.Module):
    def __init__(self, num_classes = 40):
        super(Point_dec, self).__init__()
        bw = 0.05
        self.fp4 = PointDeconv(nsample=16, in_channel=512, mlp=[512,512], bandwidth = 8*bw)
        self.fp3 = PointDeconv(nsample=16, in_channel=512, mlp=[256,256], bandwidth = 4*bw)
    
    def forward(self, l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points, l4_xyz, l4_points):
        print()
        print('sa dec 4 (sa 4) output:')
        print('l4_xyz:', l4_xyz.shape)
        print('l4_points:', l4_points.shape)
        l3_points = self.fp4(l4_xyz, l4_points, l3_xyz, l3_points)
        print()
        print('sa dec 4 (fp 4) output:')
        print('l3_points:', l3_points.shape)
        l2_points = self.fp3(l3_xyz, l3_points, l2_xyz, l2_points)
        print()
        print('sa dec 3 (fp 3) output:')
        print('l2_points:', l2_points.shape)
        return l2_points
    
if __name__ == '__main__':
    import os
    import torch
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input = torch.randn((8,3,2048))
    input2 = torch.randn((8,0,2048))
    label = torch.randn(8,16)
    model = Point_enc(num_classes=40)
    dec = Point_dec()
    output, l1_xyz, l1_points, l2_xyz, l2_points, l3_xyz, l3_points = model(input)
    # output = model(input, input2)
    print(output.size())

