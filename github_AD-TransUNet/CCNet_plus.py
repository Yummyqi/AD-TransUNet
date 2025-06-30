# -*-coding: utf-8 -*-
# @Time    : 2024/6/24 16:05
# @Author  : 宋宋
# @File    : CCNet_plus.py

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
import numpy as np


# class CAM_Module(nn.Module):
#     """ Channel attention module"""
#     def __init__(self, in_dim, dropout_rate=0.1):
#         super(CAM_Module, self).__init__()
#         self.chanel_in = in_dim
#         self.gamma = Parameter(torch.zeros(1))
#         self.softmax  = Softmax(dim=-1)
#
#         self.dropout = nn.Dropout(dropout_rate)  # 添加通道dropout层
#
#     def forward(self,x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X C X C
#         """
#         m_batchsize, C, height, width = x.size()
#         proj_query = x.view(m_batchsize, C, -1)
#         proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
#         energy = torch.bmm(proj_query, proj_key)
#         energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
#         attention = self.softmax(energy_new)
#
#         # attention = self.dropout(attention)  # 在softmax之后应用通道dropout
#
#         proj_value = x.view(m_batchsize, C, -1)
#
#         out = torch.bmm(attention, proj_value)
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out

class CAM_Module(nn.Module):
    """ Channel attention module with channel dropout"""

    def __init__(self, in_dim, dropout_rate=0.1):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1) # reshape
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        # 计算energy张量中每个元素相对于其所在最后一维（例如，W维度）的最大值的差异
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # 应用通道退出
        channel_dropout_mask = self.channel_dropout(attention)
        attention = attention * channel_dropout_mask

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

    def channel_dropout(self, attention):
        """
        实现通道退出功能

        参数:
        attention: 通道注意力矩阵 (B X C X C)

        返回:
        channel_dropout_mask: 通道退出掩码 (B X C X C)
        """
        _, C, _ = attention.size() # 获取batch size和通道数C，第三个下标是多余的，因此用_表示忽略。
        channel_dropout_mask = torch.ones_like(attention) # 创建一个与attention同形状的全1张量作为通道退出掩码。

        # 计算每个通道的保留概率
        attention_sum = torch.sum(attention, dim=-1, keepdim=True) # 对每个通道的注意力值求和，得到(B X C X 1)的张量。
        p = attention / attention_sum # 求出每个通道的保留概率，即注意力值占总注意力值的比例。

        # 生成通道退出掩码
        rand_nums = torch.rand(p.size(), device=attention.device)# 生成与p同形状的随机数，范围在[0, 1)之间。
        channel_dropout_mask[rand_nums > p] = 0# 如果随机数大于保留概率，则将对应的掩码值设为0，意味着该通道被"退出"。

        return channel_dropout_mask

# class PAM_Module(nn.Module):
#     """ Position attention module"""
#     #Ref from SAGAN
#     def __init__(self, in_dim, attn_dropout=0.1):
#         super(PAM_Module, self).__init__()
#         self.chanel_in = in_dim
#
#         self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
#         self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = Parameter(torch.zeros(1))
#
#         self.softmax = Softmax(dim=-1)
#     def forward(self, x):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         m_batchsize, C,height, width = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
#         energy = torch.bmm(proj_query, proj_key) # 矩阵乘法
#         attention = self.softmax(energy)
#
#         attention = self.attn_dropout(attention)  # 应用注意力dropout
#
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, height, width)
#
#         out = self.gamma*out + x
#         return out


class PAM_Module(nn.Module):
    """ Position attention module with region dropout"""

    def __init__(self, in_dim, attn_dropout=0.1, region_dropout_rate=0.1):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # 对QKV进行逐点卷积操作（改）
        self.query_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 4, kernel_size=1)
        )
        self.key_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_dim // 4, out_channels=in_dim // 4, kernel_size=1)
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1),
            nn.ReLU()
        )
        self.gamma = Parameter(torch.zeros(1)) # 可学习参数

        # self.attn_dropout = nn.Dropout2d(attn_dropout)
        self.region_dropout_rate = region_dropout_rate

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)  # 矩阵乘法
        attention = self.softmax(energy)

        # attention = self.attn_dropout(attention)  # 应用注意力dropout

        # 应用区域退出
        attention = self.region_dropout(attention, height, width)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # print("gamma:",self.gamma)
        out = self.gamma * out + x
        return out

    def compute_region_attention(self, attention, height, width, region_size):
        num_regions_height = height // region_size
        num_regions_width = width // region_size

        region_attention = torch.zeros(attention.size(0), num_regions_height, num_regions_width, device=attention.device)

        for i in range(num_regions_height):
            for j in range(num_regions_width):
                start_h = i * region_size
                end_h = start_h + region_size
                start_w = j * region_size
                end_w = start_w + region_size

                region_attention[:, i, j] = F.adaptive_avg_pool2d(attention[:, start_h:end_h, start_w:end_w], (1, 1)).view(-1)

        return region_attention

    def region_dropout(self, attention, height, width):
        """
        实现区域退出功能

        参数:
        attention: 注意力矩阵 (B x (H*W) x (H*W))
        height, width: 特征图的高度和宽度

        返回:
        attention_out: 经过区域退出后的注意力矩阵
        """
        B, H_W, _ = attention.size()
        # region_size = int(np.sqrt(H_W))  # 假设区域大小为正方形  28 56 112
        region_size = 7 # 在特征图上划分的正方形区域的边长
        attention_out = attention.clone()

        # 添加一个很小的值,避免全0输入导致的错误
        epsilon = 1e-8

        for i in range(region_size):
            for j in range(region_size):
                # 计算当前区域的注意力平均值
                region_attn = attention[:, i * region_size:(i + 1) * region_size,
                              j * region_size:(j + 1) * region_size].mean(dim=(1, 2))

                # 检查当前区域的注意力和是否为0,如果为0,则设置为epsilon
                region_attn_sum = region_attn.sum(dim=-1, keepdim=True)
                region_attn_sum[region_attn_sum == 0] = epsilon

                # 计算当前区域的保留概率
                region_prob = region_attn / region_attn_sum

                # 生成区域退出掩码
                region_dropout_mask = torch.ones_like(region_attn)
                rand_nums = torch.rand(region_attn.size(), device=attention.device)
                region_dropout_mask[rand_nums > region_prob] = 0

                # 应用区域退出掩码
                new_region = attention[:, i * region_size:(i + 1) * region_size,
                             j * region_size:(j + 1) * region_size] * region_dropout_mask.unsqueeze(1).unsqueeze(1)
                attention_out[:, i * region_size:(i + 1) * region_size,
                j * region_size:(j + 1) * region_size] = new_region

        return attention_out

def norm(planes, mode='bn', groups=16):
    if mode == 'bn':
        return nn.BatchNorm2d(planes, momentum=0.95, eps=1e-03)
    elif mode == 'gn':
        return nn.GroupNorm(groups, planes)
    else:
        return nn.Sequential()

class CCNetPlus(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CCNetPlus,self).__init__()
        inter_channels = in_channels // 16 #512/16=32
        # inter_channels = in_channels

        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.sa = PAM_Module(inter_channels)
        self.ca = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())
        self.conv7 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
                                   nn.ReLU())

        # self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(inter_channels, out_channels, 1),
        #                            nn.ReLU())

        self.conv8 = nn.Sequential(nn.Dropout2d(0.05, False), nn.Conv2d(in_channels, out_channels, 1),
                               nn.ReLU())

    def forward(self, x):

        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        ca_feat = self.ca(feat2)
        ca_conv = self.conv52(ca_feat)
        ca_output = self.conv7(ca_conv)

        feat_sum = sa_output + ca_output
        # print("feat_sum:",feat_sum.shape)  #[3, 768, 14, 14]

        saca_output = self.conv8(feat_sum)


        return saca_output