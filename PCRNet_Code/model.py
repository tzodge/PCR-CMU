#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import torch
import numpy as np
import torch.nn as nn
from util import quat2mat
import torch.nn.functional as F
from torch.autograd import Variable
from ops.transform_functions import PCRNetTransform as transform


class PointNet(nn.Module):
    def __init__(self, emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class Pooling(torch.nn.Module):
    def __init__(self, pool_type='max'):
        self.pool_type = pool_type
        super(Pooling, self).__init__()

    def forward(self, input):
        if self.pool_type == 'max':
            return torch.max(input, 2)[0].contiguous()
        elif self.pool_type == 'avg' or self.pool_type == 'average':
            return torch.mean(input, 2).contiguous()


class iPCRNet(nn.Module):
    def __init__(self, feature_model=PointNet(emb_dims=1024), droput=0.0, pooling='max'):
        super().__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)

        self.linear = [nn.Linear(self.feature_model.emb_dims * 2, 1024), nn.ReLU(),
                       nn.Linear(1024, 1024), nn.ReLU(),
                       nn.Linear(1024, 512), nn.ReLU(),
                       nn.Linear(512, 512), nn.ReLU(),
                       nn.Linear(512, 256), nn.ReLU()]

        if droput>0.0:
            self.linear.append(nn.Dropout(droput))
        self.linear.append(nn.Linear(256,7))

        self.linear = nn.Sequential(*self.linear)

    # Single Pass Alignment Module (SPAM)
    def spam(self, template_features, source, est_R, est_t):
        batch_size = source.size(0)

        self.source_features = self.pooling(self.feature_model(source))
        y = torch.cat([template_features, self.source_features], dim=1)
        pose_7d = self.linear(y)
        pose_7d = transform.create_pose_7d(pose_7d)

        # Find current rotation and translation.
        identity = torch.eye(3).to(source).view(1,3,3).expand(batch_size, 3, 3).contiguous()
        est_R_temp = transform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
        est_t_temp = transform.get_translation(pose_7d).view(-1, 1, 3)

        # update translation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        # update rotation matrix.
        est_R = torch.bmm(est_R_temp, est_R)
        
        source = transform.quaternion_transform(source, pose_7d)      # Ps' = est_R*Ps + est_t
        return est_R, est_t, source

    def forward(self, template, source, max_iteration=8):
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()         # (Bx3x3)
        est_t = torch.zeros(1,3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()     # (Bx1x3)
        template_features = self.pooling(self.feature_model(template))

        if max_iteration == 1:
            est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
        else:
            for i in range(max_iteration):
                est_R, est_t, source = self.spam(template_features, source, est_R, est_t)

        result = {'est_R': est_R,               # source -> template
                  'est_t': est_t,               # source -> template
                  'est_T': transform.convert2transformation(est_R, est_t),          # source -> template
                  'r': template_features - self.source_features,
                  'transformed_source': source}
        return result


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        emb_dims = args.emb_dims
        self.emb_dims = emb_dims
        self.nn = nn.Sequential(nn.Linear(emb_dims * 2, emb_dims // 2),
                                nn.BatchNorm1d(emb_dims // 2),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 2, emb_dims // 4),
                                nn.BatchNorm1d(emb_dims // 4),
                                nn.ReLU(),
                                nn.Linear(emb_dims // 4, emb_dims // 8),
                                nn.BatchNorm1d(emb_dims // 8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(emb_dims // 8, 4)
        self.proj_trans = nn.Linear(emb_dims // 8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.emb_dims = args.emb_dims
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size = src.size(0)

        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        raw_scores = scores
        scores = torch.softmax(scores, dim=2)

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())

        src_centered = src - src.mean(dim=2, keepdim=True)

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous())

        U, S, V = [], [], []
        R = []

        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0).contiguous())
            r_det = torch.det(r)
            if r_det < 0:
                u, s, v = torch.svd(H[i])
                v = torch.matmul(v, self.reflect)
                r = torch.matmul(v, u.transpose(1, 0).contiguous())
                # r = r * self.reflect
            R.append(r)

            U.append(u)
            S.append(s)
            V.append(v)

        U = torch.stack(U, dim=0)
        V = torch.stack(V, dim=0)
        S = torch.stack(S, dim=0)
        R = torch.stack(R, dim=0)

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        return R, t.view(batch_size, 3), raw_scores


class PCRNet_corr(nn.Module):
    def __init__(self, args):
        super(PCRNet_corr, self).__init__()
        self.emb_dims = args.emb_dims
        self.cycle = args.cycle
        self.emb_nn = PointNet(emb_dims=self.emb_dims)

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.emb_nn(src)
        tgt_embedding = self.emb_nn(tgt)

        rotation_ab, translation_ab, raw_scores = self.head(src_embedding, tgt_embedding, src, tgt)

        if self.cycle:
            rotation_ba, translation_ba, _ = self.head(tgt_embedding, src_embedding, tgt, src)
        else:
            rotation_ba = rotation_ab.transpose(2, 1).contiguous()
            translation_ba = -torch.matmul(rotation_ba, translation_ab.unsqueeze(2)).squeeze(2)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, raw_scores
