import numpy as np
from piqa import SSIM, LPIPS, HaarPSI
from torch import nn
import torch
import lpips

['ExpandNetLoss', 'HaarLoss', 'LPIPSLoss', 'LPIPSLoss_L2', 'SSIMLoss']

def get_loss_func(chosen_loss_name):
    """
    Grab the corresponding loss from below. Valid options: 
    ['ExpandNetLoss', 'HaarLoss', 'LPIPSLoss', 'LPIPSLoss_L1', 'SSIMLoss']

    """
    if chosen_loss_name == 'ExpandNetLoss':
        print("Using ExpandNetLoss")
        return ExpandNetLoss()
    elif chosen_loss_name == 'HaarLoss':
        print("Using HaarLoss")
        return HaarLoss()
    elif chosen_loss_name == 'LPIPSLoss':
        print("Using LPIPSLoss")
        return LPIPSLoss()
    elif chosen_loss_name == 'LPIPSLoss_L1':
        print("Using LPIPSLoss_L1")
        return LPIPSLoss_L1()
    elif chosen_loss_name == 'SSIMLoss':
        print("Using SSIMLoss")
        return SSIMLoss()
    else:
        print("[ERROR] Specified loss not found...")

    return

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
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.similarity = lpips.LPIPS(net='vgg').cuda()
        # self.similarity = LPIPS(network='vgg').cuda()
        # self.l2_loss = nn.MSELoss()
        # self.loss_lambda = loss_lambda

    def forward(self, x, y):
        # norm to -1, 1
        sim = self.similarity((x / 0.5 - 1).float(), (y / 0.5 - 1).float()).squeeze().mean()
        return sim

class LPIPSLoss_L1(nn.Module):
    def __init__(self, loss_lambda=0.5):
        super(LPIPSLoss_L1, self).__init__()
        self.similarity = lpips.LPIPS(net='vgg').cuda()
        self.l1_loss = nn.L1Loss()
        # self.similarity = LPIPS(network='vgg').cuda()
        # self.l2_loss = nn.MSELoss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        # sim = self.similarity(x.float(), y.float())
        sim = self.similarity((x / 0.5 - 1).float(), (y / 0.5 - 1).float()).squeeze().mean()
        return (1 - self.loss_lambda) * self.l1_loss(x, y) + self.loss_lambda * (sim)

class SSIMLoss(nn.Module):
    def __init__(self, loss_lambda=0.5):
        super(SSIMLoss, self).__init__()
        self.similarity = SSIM().cuda()
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return (1 - self.loss_lambda) * self.l1_loss(x, y) + self.loss_lambda * (1-sim)