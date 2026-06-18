#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.pc_nbv_model_utils import Feature_Extraction, SelfAttention, normalize_point_batch
from pfrl.initializers import init_chainer_default
#use KAN in this file


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.feature_extraction = Feature_Extraction()
        self.bn1 = nn.BatchNorm1d(264)

        self.position_emb = config['positionemb']
        self.weighted_pc = config['weightedpc']
        self.recurrent = config['recurrent']

        if self.weighted_pc:
            attention_in_channel = 264 + 264 + 128+1 #264+264+3+1
        else:
            attention_in_channel = 264 + 264 + 128#+ view_dim

        recurrent_in_chan = 128 + 60 if self.position_emb else 128 + 3

        self.attention_unit = SelfAttention(attention_in_channel)
        self.bn2 = nn.BatchNorm1d(attention_in_channel)

        self.conv1 = nn.Conv1d(attention_in_channel, 1024, 1)
        self.conv2 = nn.Conv1d(1024, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(1024)

        if self.position_emb:
            self.seq1 = init_chainer_default(nn.Sequential(
                            nn.Linear(60, 128),
                            nn.ReLU(),
                            nn.Linear(128, 128), #(B, 1, 64)
                            nn.ReLU(),       
                            ))
        else:
            self.seq1 = init_chainer_default(nn.Sequential(
                            nn.Linear(3, 32),
                            nn.ReLU(),
                            nn.Linear(32, 64),
                            nn.ReLU(),
                            nn.Linear(64, 128), #(B, 1, 64)
                            nn.ReLU(),
                            nn.Linear(128, 128),
                            nn.ReLU(),
                            ))
        if self.recurrent:
            self.seq2_ = init_chainer_default(nn.Sequential(
                            nn.Linear(recurrent_in_chan, 128), #(B, 1, 64)
                            nn.ReLU()))
            
            self.rnn = nn.GRU(input_size= 128, hidden_size=128, batch_first = True)
            self.seq2 = init_chainer_default(nn.Sequential(
                            nn.Linear(128, 128),
                            nn.ReLU()))
        else:
            self.seq2 = init_chainer_default(nn.Sequential(
                            nn.Linear(recurrent_in_chan, 128), #(B, 1, 64)
                            nn.ReLU(),
                            nn.Linear(128, 128), #(B, 1, 64)
                            nn.ReLU()))

    def forward(self, pcs, view_pose, recurrent_view):
        pcs = normalize_point_batch(pcs)      #(B, 3, N)
        n = pcs.size()[2]

        x = self.feature_extraction(pcs)         #(B, 264, N)
        x = self.bn1(x)
        g = torch.max(x, dim=2, keepdim=True)[0]    #(B, 264, 1)
        g = g.repeat(1, 1, n)   
                           
        out_move = self.seq1(view_pose)  #256
        out_move = torch.cat([out_move, view_pose], dim=-1)

        if self.recurrent:
            out_move = self.seq2_(out_move)
            out_move, recurrent_view_tp1 = self.rnn(out_move, recurrent_view) #128
            view_out = self.seq2(recurrent_view_tp1)  #32
        else:
            view_out = self.seq2(out_move)  #32
        vi = view_out.squeeze(0).unsqueeze(2).repeat(1, 1, n)

        # if self.weighted_pc:
        #     weights = weights.unsqueeze(1)
        #     x = torch.cat([x, weights, g, vi], dim = 1)          #(B, 532, N)   544 = 264 + 264 + 32+ 64+1
        # else:
        x = torch.cat([x, g, vi], dim = 1)  #531 = 264 + 264 + 128 
        
        x = self.attention_unit(F.relu(x))          #(B, 534, N)
        x = self.bn2(x)
        x = F.relu(self.bn3(self.conv1(x)))         #(B, 1024, N)
        x = self.bn4(self.conv2(x))                 #(B, 1024, N)
        v = torch.max(x, dim = -1)[0]               #(B, 1024)
        return v, recurrent_view_tp1

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.linear1 = init_chainer_default(nn.Linear(1024, 1024))  ## this need to remove
        self.linear2 = init_chainer_default(nn.Linear(1024, 512))
        self.linear3 = init_chainer_default(nn.Linear(512, 256))
        self.linear4 = init_chainer_default(nn.Linear(256, config['num_actions']))

        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.linear1(x)))       #(B, 1024)
        x = F.relu(self.bn2(self.linear2(x)))       #(B, 512)
        x = F.relu(self.bn3(self.linear3(x)))       #(B, 256)
        v = self.linear4(x) 
        return v

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, pcs, view_pose, recurrent_view):
        # pcs = weighted_pcs[:, 0:3, :]
        # weights = weighted_pcs[:, 3, :]
        pcs = pcs[:, 0:3,:]
        x, recurrent_view_tp1 = self.encoder(pcs, view_pose, recurrent_view)
        v = self.decoder(x)
        return v, recurrent_view_tp1

if __name__ == "__main__":

    config = {'positionemb': True, 'weightedpc': False, 'recurrent': True, 'num_actions': 8}

    ae = AutoEncoder(config).cuda()
    weighted_pcs = torch.rand(2, 7, 512).cuda()
    view_pose = torch.rand(2, 1, 60).cuda()
    recurrent_view = torch.rand(1, 2, 128).cuda()
    
    v, recurrent_view_tp1 = ae(weighted_pcs, view_pose, recurrent_view)
    
  
