import numpy as np
from piqa import SSIM, LPIPS, HaarPSI
from torch import nn
import torch


class ExpandNetLoss(nn.Module):
    def __init__(self, loss_lambda=5):
        super(ExpandNetLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        cosine_term = (1 - self.similarity(x, y)).mean()
        return self.l1_loss(x, y) + self.loss_lambda * cosine_term

class HaarLoss(nn.Module):
    def __init__(self, loss_lambda=0.7):
        super(HaarLoss, self).__init__()
        self.similarity = HaarPSI().cuda()
        self.l2_loss = nn.MSELoss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return (1 - self.loss_lambda) * self.l2_loss(x, y) + self.loss_lambda * (1 - sim)

class LPIPSLoss(nn.Module):
    def __init__(self, loss_lambda=0.8):
        super(LPIPSLoss, self).__init__()
        self.similarity = LPIPS().cuda()
        self.l2_loss = nn.MSELoss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return (1 - self.loss_lambda) * self.l2_loss(x, y) + self.loss_lambda * (sim)

class SSIMLoss(nn.Module):
    def __init__(self, loss_lambda=0.5):
        super(SSIMLoss, self).__init__()
        self.similarity = SSIM().cuda()
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return (1 - self.loss_lambda) * self.l1_loss(x, y) + self.loss_lambda * (1-sim)