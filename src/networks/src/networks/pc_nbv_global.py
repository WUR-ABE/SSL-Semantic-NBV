#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.pc_nbv_model_utils import Feature_Extraction, SelfAttention, normalize_point_batch


class Encoder_xyz_viewstate(nn.Module):
    def __init__(self):
        super(Encoder_xyz_viewstate, self).__init__()
        self.feature_extraction = Feature_Extraction(in_channels=7)
        self.bn1 = nn.BatchNorm1d(264)
        
        # Adaptive feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(264 + 264, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        
        # Attention mechanism
        self.attention = SelfAttention(1024)
        
    def forward(self, inputs):
        inputs = normalize_point_batch(inputs)  # (B, 4, N)
        n = inputs.size()[2]
        
        # Point cloud features
        x = self.feature_extraction(inputs)      # (B, 264, N)
        x = self.bn1(x)
        
        # Global context
        g = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, n)  # (B, 264, N)
        combined = torch.cat([x, g], dim=1)  # (B, 264+264+32, N)

        fused = self.feature_fusion(combined)              # (B, 1024, N)
        
        # Attention and aggregation
        attended = self.attention(fused)                   # (B, 1024, N)
        v = torch.max(attended, dim=2)[0]                  # (B, 1024)
        return v
    
class Encoder_xyz_viewstate_work2(nn.Module):
    def __init__(self):
        super(Encoder_xyz_viewstate_work2, self).__init__()
        self.feature_extraction = Feature_Extraction(in_channels=7)
        self.bn1 = nn.BatchNorm1d(264)
        
        # Adaptive feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(264 + 264 + 81, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.BatchNorm1d(1024)
        )
        
        # Attention mechanism
        self.attention = SelfAttention(1024)
        
    def forward(self, inputs, viewstate):
        inputs = normalize_point_batch(inputs)  # (B, 4, N)
        n = inputs.size()[2]
        
        # Point cloud features
        x = self.feature_extraction(inputs)      # (B, 264, N)
        x = self.bn1(x)
        
        viewstate = viewstate.unsqueeze(2).repeat(1, 1, n)
        # Global context
        g = torch.max(x, dim=2, keepdim=True)[0].repeat(1, 1, n)  # (B, 264, N)
        combined = torch.cat([x, g, viewstate], dim=1)  # (B, 264+264+81, N)

        fused = self.feature_fusion(combined)              # (B, 1024, N)
        
        # Attention and aggregation
        attended = self.attention(fused)                   # (B, 1024, N)
        v = torch.max(attended, dim=2)[0]                  # (B, 1024)
        return v

# class Encoder_RGB(nn.Module):
#     def __init__(self):
#         super(Encoder_RGB, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # (64,64) -> (32,32)
#         )
        
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # (32,32) -> (16,16)
#         )

#         self.conv3 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))  # (16,16) -> (8,8)
#         )

#         self.linear = nn.Sequential(
#             nn.Linear(1344, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#         )

#         self.gru = nn.GRU(input_size=512, hidden_size=128, num_layers=1, batch_first=True)

#     def forward(self, rgb, recurrent_rgb):
#         rgb_32 = self.conv1(rgb)                     # (B, 16, 64, 64)
#         rgb_16 = self.conv2(rgb_32)                    # (B, 32, 32, 32)
#         rgb_8 = self.conv3(rgb_16)                    # (B, 64, 16, 16)

#         max_rgb_32  = torch.max(rgb_32, dim=1, keepdim=True)[0] # (B, 1, 32, 32)
#         max_rgb_16  = torch.max(rgb_16, dim=1, keepdim=True)[0] # (B, 1, 16, 16)
#         max_rgb_8  = torch.max(rgb_8, dim=1, keepdim=True)[0] # (B, 1, 8, 8)

#         max_rgb_32 = max_rgb_32.view(max_rgb_32.size(0), 1,  -1) # (B, 1, 1024)
#         max_rgb_16 = max_rgb_16.view(max_rgb_16.size(0), 1,  -1) # (B, 1, 256)
#         max_rgb_8 = max_rgb_8.view(max_rgb_8.size(0), 1,  -1) # (B, 1, 64)

#         fused_rgb = torch.cat([max_rgb_32, max_rgb_16, max_rgb_8], dim=-1) # shape (B, 1, 1024+256+64)
#         rgb_features = self.linear(fused_rgb) # shape (B, 1, 512)

#         _, recurrent_rgb_tp1 = self.gru(rgb_features, recurrent_rgb) # shape (B, 1, 128)
#         recurrent_rgb_tp1 = recurrent_rgb_tp1.permute(1, 0, 2) # shape (B, 1, 128)
#         return rgb_features, recurrent_rgb_tp1
    
# class Decoder_RGB(nn.Module):
#     def __init__(self):
#         super(Decoder_RGB, self).__init__()
        
#         # View embedding network
#         self.view_embedding = nn.Sequential(
#             nn.Linear(40, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128)
#         )
        
#         # Feature fusion network
#         self.linear = nn.Sequential(
#             nn.Linear(640, 1024),  # Combine view embedding and recurrent RGB
#             nn.ReLU(),
#         )

#         self.conv1 = nn.ConvTranspose2d(1024, 128, 5, stride=2)
#         self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
#         self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
#         self.conv4_m = nn.ConvTranspose2d(32, 3, 6, stride=2)
#         self.conv4_s = nn.ConvTranspose2d(32, 3, 6, stride=2)

#         self.fc_combined = nn.Linear(128+128, 1024)
    
#         # Final convolution to generate RGB channels
#         self.conv2d = nn.Conv2d(1, 3, kernel_size=3, padding=1)
    
#     def forward(self, recurrent_rgb_tp1, num_views=40):
#         recurrent_rgb_tp1 = recurrent_rgb_tp1.squeeze(1)
        
#         # Create one-hot encoding for each view
#         one_hot_views = torch.eye(num_views).unsqueeze(0).repeat(recurrent_rgb_tp1.size(0), 1, 1).to(recurrent_rgb_tp1.device)  # (B, num_views, num_views)
        
#         rgbs = []
#         for i in range(num_views):
#             one_hot_view = one_hot_views[:, i, :]
#             view_embedding = self.view_embedding(one_hot_view)

#             combined = torch.cat([view_embedding, recurrent_rgb_tp1], dim=-1) # shape (B, 128+128)

#             # features = self.linear(combined) # shape (B, 1024)
#             # m = self.fc_posterior_m(features)
#             # s = F.softplus(self.fc_posterior_s(features)) + 1e-1

#             # features = m + torch.randn_like(m) * s

#             # feature_combined = torch.cat([features, recurrent_rgb_tp1], dim=-1) # shape (B, 1024+128)
#             # feature_combined = self.fc_combined(feature_combined) # shape (B, 1024)
            
#             combined = self.fc_combined(combined) # shape (B, 1024)

#             features = combined.view(-1, 1024, 1, 1)

#             rgb = F.relu(self.conv1(features))
#             rgb = F.relu(self.conv2(rgb))
#             rgb = F.relu(self.conv3(rgb))
#             rgb_m = F.relu(self.conv4_m(rgb))
#             rgb_s = F.relu(self.conv4_s(rgb))

#             rgb = rgb_m + torch.randn_like(rgb_m) * rgb_s
#             rgbs.append(rgb)
            
#         return torch.stack(rgbs, dim=1)

class PositionEmbedding(nn.Module):
    def __init__(self, num_views, output_dim=81, num_freqs=10):
        super(PositionEmbedding, self).__init__()
        self.num_views = num_views
        self.output_dim = output_dim
        self.num_freqs = num_freqs
        
        # NeRF-style positional encoding dimension
        # Each frequency contributes 2 dimensions (sin and cos)
        self.pos_encoding_dim = num_freqs * 2
        
        # Final projection to output dimension
        self.projection = nn.Sequential(
            nn.Linear(self.pos_encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        self.fusion = nn.Sequential(nn.Linear((self.pos_encoding_dim+1), 8), # new
                                    nn.ReLU(),
                                    nn.Linear(8, 4),
                                    nn.ReLU(),
                                    nn.Linear(4, 1)
                                    )
        
    def positional_encoding(self, positions):
        """
        Apply NeRF-style positional encoding to view indices
        Args:
            positions: (B, num_views) - view indices or normalized positions
        Returns:
            encoded: (B, num_views, pos_encoding_dim)
        """
        batch_size, num_views = positions.shape
        
        # Create frequency bands
        freq_bands = torch.pow(2.0, torch.linspace(0, self.num_freqs-1, self.num_freqs)).to(positions.device)
        
        # Apply positional encoding
        encoded = []
        encoded.append(positions.unsqueeze(-1)) # new
        for freq in freq_bands:
            # Sin and cos for each frequency
            encoded.append(torch.sin(positions.unsqueeze(-1) * freq))
            encoded.append(torch.cos(positions.unsqueeze(-1) * freq))
        
        # Concatenate all encodings
        encoded = torch.cat(encoded, dim=-1)  # (B, num_views, pos_encoding_dim+1)
        
        return encoded
    
    def forward(self, viewstate):
        """
        Args:
            viewstate: (B, num_views) - binary or continuous viewstate
        Returns:
            position_embeddings: (B, 81)
        """
        batch_size, num_views = viewstate.shape
        
        # Create view indices (0, 1, 2, ..., num_views-1)
        view_indices = torch.arange(num_views, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1).to(viewstate.device)
        
        # Normalize view indices to [0, 1]
        normalized_indices = view_indices / (num_views - 1)
        
        # Apply NeRF-style positional encoding
        pos_encoded = self.positional_encoding(normalized_indices)  # (B, num_views, pos_encoding_dim+1)
        
        # Weight by viewstate activations
        weighted_encoded = pos_encoded * viewstate.unsqueeze(-1)  # (B, num_views, pos_encoding_dim+1)
        
        # fuse the weighted_encoded and pos_encoded to get B, number_views, 1
        fused = self.fusion(weighted_encoded).squeeze(-1) # new
        
        # Aggregate across views (sum pooling)
        #aggregated = torch.sum(weighted_encoded, dim=-1)  # (B, num_views)
        
        # # Project to output dimension
        # position_output = self.projection(aggregated)  # (B, 81)
        # print(position_output.shape)
        
        return fused

class Decoder(nn.Module):
    def __init__(self, number_views=33):
        super(Decoder, self).__init__()
        self.dropout = False

        self.linear1 = nn.Sequential(nn.Linear(1024+number_views, 1024),
                                      nn.BatchNorm1d(1024),
                                      nn.ReLU(),
                                      )
        self.linear2 = nn.Sequential(nn.Linear(1024+number_views, 512),
                                     nn.BatchNorm1d(512),
                                     nn.ReLU(),
                                     )
        self.linear3 = nn.Sequential(nn.Linear(512+number_views, 256),
                                     nn.BatchNorm1d(256),
                                     nn.ReLU(),
                                     )

        self.linear4 = nn.Sequential(
            nn.Linear(256+number_views, 128),
            nn.ReLU(),
            nn.Linear(128, number_views)
        )

        self.position_embedding = PositionEmbedding(number_views, output_dim=81)

    def forward(self, x, viewstate):
        viewstate = self.position_embedding(viewstate)
        x = torch.cat([x, viewstate], dim=1)
        x = self.linear1(x)       #(B, 1024)
        x = torch.cat([x, viewstate], dim=1)
        x = self.linear2(x)       #(B, 512)
        x = torch.cat([x, viewstate], dim=1)
        x = self.linear3(x)       #(B, 256)
        x = torch.cat([x, viewstate], dim=1)
        v = self.linear4(x) 
        return v

class Decoder_work2(nn.Module):
    def __init__(self, number_views=33):
        super(Decoder_work2, self).__init__()

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
    
class AutoEncoder(nn.Module):
    def __init__(self, number_views=40, planner='ssl_semantic_nbv'):
        super(AutoEncoder, self).__init__()
        self.number_views = number_views

        self.planner = planner
        
        if 'NOPE' in planner.split('_'):
            self.encoder_xyz_viewstate_work2 = Encoder_xyz_viewstate_work2()
            self.decoder_work2 = Decoder_work2(number_views)
        else:
            self.encoder_xyz_viewstate = Encoder_xyz_viewstate()
            self.decoder = Decoder(number_views)

        # self.init_weights()
        
    def forward(self, x, viewstate):
        if 'NOPE' in self.planner.split('_'):
            x = self.encoder_xyz_viewstate_work2(x, viewstate) 
            pre_ig = self.decoder_work2(x)
        else:
            x = self.encoder_xyz_viewstate(x) 
            pre_ig = self.decoder(x, viewstate)
        return pre_ig
    
    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             if isinstance(m, nn.Sequential) or hasattr(m, 'out_features') and m.out_features == self.number_views:
    #                 nn.init.xavier_uniform_(m.weight)  # output layer
    #             else:
    #                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # ReLU layers
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if hasattr(m, 'out_features') and m.out_features == self.number_views:
                    # Output layer - very small weights for stable initial predictions
                    nn.init.normal_(m.weight, mean=0.0, std=0.001)
                else:
                    # Hidden layers - small weights around 0
                    nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                # For conv layers, also use small initialization
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

if __name__ == "__main__":
    ae = AutoEncoder(54).cuda()
    ae.train()
    x = torch.rand(2, 7, 512).cuda()
    viewstate = torch.rand(2, 54).cuda()
    with torch.no_grad():
        # enable dropout        
        for i in range(2):
            pre_ig = ae(x, viewstate)
            print(pre_ig)
    




        
