import torch
from torch import nn
from os import path as osp
import shutil
import logging
# from src.models.UNet import get_unet


# Define Network Blocks
def convolution_block(in_dim, out_dim):
    return nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=1, padding=1)

# def convolution_block_resnet(in_dim, out_dim):
#     return nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=2, padding=1)

def transpose_convolution_block(in_dim, out_dim):
    return torch.nn.ConvTranspose2d(in_dim,out_dim,kernel_size=3, stride=2, padding=1,output_padding=1)

def contract_block(in_dim,out_dim,activation):
    block = nn.Sequential(convolution_block(in_dim, out_dim),
                            nn.BatchNorm2d(out_dim),
                            activation,
                            convolution_block(out_dim, out_dim),
                            nn.BatchNorm2d(out_dim),
                            activation)
    return block

def bottom_block(in_dim, out_dim, activation):
    bottom = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 convolution_block(out_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 transpose_convolution_block(out_dim, in_dim),
                                 nn.BatchNorm2d(in_dim),
                                 activation)
    return bottom

def bottom_block_resnet(in_dim, out_dim, activation):
    bottom = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 convolution_block(out_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 transpose_convolution_block(out_dim, in_dim//2),
                                 nn.BatchNorm2d(in_dim//2),
                                 activation)
    return bottom


def expand_block(in_dim, out_dim, activation):
    expand = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 convolution_block(out_dim, out_dim),
                                 nn.BatchNorm2d(out_dim),
                                 activation,
                                 transpose_convolution_block(out_dim, out_dim//2),
                                 nn.BatchNorm2d(out_dim//2),
                                 activation)
    return expand

def top_block(in_dim, out_dim, activation):
    top = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                              nn.BatchNorm2d(out_dim),
                              activation,
                              convolution_block(out_dim, out_dim),
                              # nn.BatchNorm2d(out_dim),
                              # activation,
                              # convolution_block(out_dim,NUM_CLASSES),
                              # nn.BatchNorm2d(NUM_CLASSES),
                              # activation
                              )
    return top

def top_block_resnet(in_dim, out_dim, resnet_out_dim, activation):
    top = torch.nn.Sequential(convolution_block(in_dim, out_dim),
                              nn.BatchNorm2d(out_dim),
                              activation,
                              transpose_convolution_block(out_dim, out_dim),
                              nn.BatchNorm2d(out_dim),
                              activation,
                              convolution_block(out_dim, out_dim),
                            #   nn.BatchNorm2d(out_dim),
                            #   activation,
                            #   convolution_block(out_dim,resnet_out_dim),
                            #   nn.BatchNorm2d(resnet_out_dim),
                            #   activation
                              )
    return top


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_best_model(model, epoch, performance, logdir):
    """
    Save the best model (so-far). 
    """
    save_dict = {
        'epoch': epoch,
        'unet_state_dict': model.state_dict(),
        'performance': performance
    }

    filename = osp.join(logdir, 'model_best.pth.tar')
    torch.save(save_dict, filename)

    with open(osp.join(logdir, 'best.txt'), 'w') as f:
        f.write(str(float(performance)))


def save_last_model(model, epoch, performance, logdir):
    """
    Save the last model in training loop.
    """

    save_dict = {
        'epoch': epoch,
        'unet_state_dict': model.state_dict(),
        'performance': performance
    }

    filename = osp.join(logdir, 'model_last.pth.tar')
    torch.save(save_dict, filename)
