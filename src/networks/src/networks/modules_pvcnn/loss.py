import torch.nn as nn

import networks.modules_pvcnn.functional as F

__all__ = ['KLLoss']


class KLLoss(nn.Module):
    def forward(self, x, y):
        return F.kl_loss(x, y)
