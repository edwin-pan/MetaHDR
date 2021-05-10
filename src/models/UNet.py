import torch
from torch import nn
import torchvision.models as models

from src.models.utils import convolution_block, transpose_convolution_block, \
    contract_block, bottom_block, expand_block, top_block, bottom_block_resnet, top_block_resnet
    
class UNet(nn.Module):
    def __init__(self, in_size, out_size, num_filters):
        super(UNet, self).__init__()
        activation_function = torch.nn.ReLU(inplace=True)

        self.contract1 = contract_block(in_size, num_filters, activation_function)
        self.contract2 = contract_block(num_filters, 2*num_filters, activation_function)
        self.contract3 = contract_block(2*num_filters, 4*num_filters, activation_function)
        self.contract4 = contract_block(4*num_filters, 8*num_filters, activation_function)
        self.bottom = bottom_block(8*num_filters, 16*num_filters, activation_function)

        # Convolutional Layers -- Expanding
        self.expand4 = expand_block(16*num_filters, 8*num_filters, activation_function)
        self.expand3 = expand_block(8*num_filters, 4*num_filters, activation_function)
        self.expand2 = expand_block(4*num_filters, 2*num_filters, activation_function)
        self.top = top_block(2*num_filters, out_size, activation_function)

        # Define Pooling Operator
        self.pool = torch.nn.MaxPool2d((2,2), stride=2)

    def get_alpha_mask(self, x, threshold=0.12):
        # Highlight mask
        alpha, _ = torch.max(x, dim=1) # Find max, per channel
        alpha = torch.minimum(torch.ones(1), torch.maximum(torch.zeros(1), alpha-1.0 + threshold)/threshold)
        alpha = alpha.reshape((-1, 1, x.shape[2], x.shape[3]))
        alpha = alpha.expand(-1, 3, -1, -1)
        return alpha

    def forward(self, x):
        # Apply Contracting Layers
        c1 = self.contract1(x)
        p1 = self.pool(c1)
        c2 = self.contract2(p1)
        p2 = self.pool(c2)
        c3 = self.contract3(p2)
        p3 = self.pool(c3)
        c4 = self.contract4(p3)
        p4 = self.pool(c4)
        bb = self.bottom(p4)

        # Apply Expanding Layers
        f4 = torch.cat([bb,c4],1)
        e4 = self.expand4(f4)
        f3 = torch.cat([e4,c3],1)
        e3 = self.expand3(f3)
        f2 = torch.cat([e3,c2],1)
        e2 = self.expand2(f2)
        f1 = torch.cat([e2,c1],1)
        out = self.top(f1)
        # out = torch.sigmoid(out)
        res = torch.relu(out)

        rec = res*self.get_alpha_mask(x) + x
        return rec


class Resnet(nn.Module):
    def __init__(self, in_size, out_size):
        super(Resnet, self).__init__()

        # Define Activation Function
        activation_function = torch.nn.ReLU(inplace=True)

        # Load pretrained ResNet Model
        self.resnet18 = models.resnet18(pretrained=True)

        # Set gradients to false
        for param in self.resnet18.parameters():
            param.requires_grad = False

        # Replace last fc layer
        num_feats = self.resnet18.fc.in_features

        # Define Contraction layers
        self.bottom = bottom_block_resnet(512, 1024, activation_function)

        # Define Expansion layers
        self.expand4 = expand_block(512, 256, activation_function)
        self.expand3 = expand_block(256, 128, activation_function)
        self.expand2 = expand_block(128, 64, activation_function)
        self.expand1 = expand_block(64, 32, activation_function)
        self.expand0 = expand_block(32, 16, activation_function)
        self.top = top_block_resnet(96, out_size, out_size, activation_function)

        # Define Pooling Operator
        self.pool = torch.nn.MaxPool2d((2,2), stride=2)

    def forward(self, x):
        # Contract
        c0 = self.resnet18.relu(self.resnet18.bn1(self.resnet18.conv1(x)))
        p0 = self.resnet18.maxpool(c0)
        c1 = self.resnet18.layer1(p0)
        c2 = self.resnet18.layer2(c1)
        c3 = self.resnet18.layer3(c2)
        c4 = self.resnet18.layer4(c3)
        bb = self.bottom(c4)
        # Expand
        f4 = torch.cat([bb,c3],1)
        e4 = self.expand4(f4)
        f3 = torch.cat([e4,c2],1)
        e3 = self.expand3(f3)
        f2 = torch.cat([e3,c1],1)
        e2 = self.expand2(f2)
        f1 = torch.cat([e2,c0],1)
        out = self.top(f1)

        out = torch.sigmoid(out)
        return out