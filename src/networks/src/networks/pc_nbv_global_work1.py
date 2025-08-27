#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.pc_nbv_model_utils import Feature_Extraction, SelfAttention, normalize_point_batch

class Encoder(nn.Module):
    def __init__(self, number_views):
        super(Encoder, self).__init__()
        self.feature_extraction = Feature_Extraction(in_channels=3)
        self.bn1 = nn.BatchNorm1d(264)

        in_channel = 264 + 264 + number_views
        self.attention_unit = SelfAttention(in_channel)
        self.bn2 = nn.BatchNorm1d(in_channel)

        self.conv1 = nn.Conv1d(in_channel, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)

    def forward(self, inputs, view_state):
        inputs = normalize_point_batch(inputs)      #(B, 3, N)
        n = inputs.size()[2]

        x = self.feature_extraction(inputs)         #(B, 264, N)
        x = self.bn1(x)

        g = torch.max(x, dim=2, keepdim=True)[0]    #(B, 264, 1)
        g = g.repeat(1, 1, n)                       # (B, 264, N) ????

        vi = view_state.unsqueeze(2).repeat(1, 1, n)
        x = torch.cat([x, g, vi], dim = 1)          #(B, 561, N)   561 = 264 + 264 + 144

        x = self.attention_unit(F.relu(x))          #(B, 561, N)
        x = self.bn2(x)

        x = F.relu(self.bn3(self.conv1(x)))         #(B, 1024, N)
        x = self.bn4(self.conv2(x))                 #(B, 1024, N)

        v = torch.max(x, dim = -1)[0]               #(B, 1024)

        return v

class Decoder(nn.Module):
    def __init__(self, number_views=33):
        super(Decoder, self).__init__()

        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, number_views)

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)


    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))       #(B, 1024)
        x = F.relu(self.bn2(self.linear2(x)))       #(B, 512)
        x = F.relu(self.bn3(self.linear3(x)))       #(B, 256)
        v = self.linear4(x) 
        return v

class AutoEncoderWork1(nn.Module):
    def __init__(self, number_views=33):
        super(AutoEncoderWork1, self).__init__()

        self.encoder = Encoder(number_views)
        self.decoder = Decoder(number_views)
    
    def forward(self, x, viewstate):
        x = x[:, :3, :]
        x = self.encoder(x, viewstate) 
        v = self.decoder(x)
        return v


if __name__ == "__main__":
   
    ae = AutoEncoderWork1(144).cuda()
    optimizer = torch.optim.Adam(ae.parameters())
    Loss = nn.MSELoss(reduction='sum')

    for i in range(2000):
        pcs = torch.rand(2, 3, 512).cuda()
        viewstate = torch.rand(2, 144).cuda()

        query = torch.zeros((2, 33), dtype=torch.int16).cuda()
        query[0, 0] = 1
        query[1, 1] = 1

        gt = torch.zeros((2, 33)).cuda()
        gt[0, 0] = 0.5
        gt[1, 1] = 0.6
        x, v = ae(pcs, viewstate)
        pred = v[query.type(torch.bool)]
        gt_va = gt[query.type(torch.bool)]
        loss = Loss(pred, gt_va)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
