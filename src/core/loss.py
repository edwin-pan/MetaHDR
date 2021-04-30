import numpy as np
from piqa import SSIM, LPIPS, HaarPSI
from torch import nn
from torchvision import models
import torch
import lpips

ACCEPTABLE_LOSS_FUNCTIONS = ['ExpandNetLoss', 'HaarLoss', 'LPIPSLoss', 'LPIPSLoss_L2', 'SSIMLoss', 'VGGLoss']

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
    elif chosen_loss_name == 'VGGLoss':
        print("Using VGGLoss")
        return VGGLoss()
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
    def __init__(self, loss_lambda=0.5):
        super(HaarLoss, self).__init__()
        self.similarity = HaarPSI().cuda()
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return (1 - self.loss_lambda) * self.l1_loss(x, y) + self.loss_lambda * (1 - sim)

class LPIPSLoss(nn.Module):
    def __init__(self):
        super(LPIPSLoss, self).__init__()
        self.similarity = LPIPS(network='vgg').cuda()

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return sim

class LPIPSLoss_L1(nn.Module):
    def __init__(self, scale=10):
        super(LPIPSLoss_L1, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.similarity = LPIPS(network='vgg').cuda()
        self.scale = scale

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return self.l1_loss(x, y) + self.scale * (sim)

class SSIMLoss(nn.Module):
    def __init__(self, loss_lambda=0.5):
        super(SSIMLoss, self).__init__()
        self.similarity = SSIM().cuda()
        self.l1_loss = nn.L1Loss()
        self.loss_lambda = loss_lambda

    def forward(self, x, y):
        sim = self.similarity(x.float(), y.float())
        return (1 - self.loss_lambda) * self.l1_loss(x, y) + self.loss_lambda * (1-sim)


# Taken (with graditude), from https://github.com/mukulkhanna/FHDR/blob/master/vgg.py
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# Taken (with graditude), from https://github.com/mukulkhanna/FHDR/blob/master/vgg.py
class VGGLoss(nn.Module):
    def __init__(self, scale=10):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.l1_loss = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.scale = scale

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x.float()), self.vgg(y.float())
        sim = 0
        for i in range(len(x_vgg)):
            sim += self.weights[i] * self.l1_loss(x_vgg[i], y_vgg[i].detach())
        return self.l1_loss(x,y) + self.scale*sim