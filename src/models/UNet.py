import torch
from torch import nn
import torchvision.models as models

from src.models.utils import convolution_block, transpose_convolution_block, \
    contract_block, bottom_block, expand_block, top_block
    
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
        out = torch.sigmoid(out)

        return out


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
        self.bottom = bottom_block(512, 1024, activation_function)

        # Define Expansion layers
        self.expand4 = expand_block(512, 256, activation_function)
        self.expand3 = expand_block(256, 128, activation_function)
        self.expand2 = expand_block(128, 64, activation_function)
        self.expand1 = expand_block(64, 32, activation_function)
        self.expand0 = expand_block(32, 16, activation_function)
        self.top = top_block(96, 64, activation_function)

        # Define Pooling Operator
        self.pool = torch.nn.MaxPool2d((2,2), stride=2)

    def forward(self, x):
        print("[Debug] x.shape: ", x.shape) # [Debug] x.shape:  torch.Size([32, 3, 224, 288])
        # Contract
        c0 = self.resnet18.relu(self.resnet18.bn1(self.resnet18.conv1(x)))
        print("[Debug] c0.shape: ", c0.shape) # [Debug] c0.shape:  torch.Size([32, 64, 112, 144])
        p0 = self.resnet18.maxpool(c0)

        c1 = self.resnet18.layer1(p0)
        print("[Debug] c1.shape: ", c1.shape) # [Debug] c1.shape:  torch.Size([32, 64, 56, 72])

        c2 = self.resnet18.layer2(c1)
        print("[Debug] c2.shape: ", c2.shape) # [Debug] c2.shape:  torch.Size([32, 128, 28, 36])

        c3 = self.resnet18.layer3(c2)
        print("[Debug] c3.shape: ", c3.shape) # [Debug] c3.shape:  torch.Size([32, 256, 14, 18])

        c4 = self.resnet18.layer4(c3)
        print("[Debug] c4.shape: ", c4.shape) # [Debug] c4.shape:  torch.Size([32, 512, 7, 9])

        bb = self.bottom(c4)
        print("[Debug] bb.shape: ", bb.shape) # [Debug] bb.shape:  torch.Size([32, 512, 14, 18]


        # Expand
        f4 = torch.cat([bb,c3],1)
        print("[Debug] f4.shape: ", f4.shape) # [Debug] f4.shape:  torch.Size([32, 512, 14, 18])

        e4 = self.expand4(f4)
        print("[Debug] e4.shape: ", e4.shape) # [Debug] e4.shape:  torch.Size([32, 256, 28, 36])

        f3 = torch.cat([e4,c2],1)
        print("[Debug] f3.shape: ", f3.shape) # [Debug] f3.shape:  torch.Size([32, 384, 28, 36])

        e3 = self.expand3(f3)
        print("[Debug] e3.shape: ", e3.shape) # [Debug] e3.shape:  torch.Size([32, 64, 56, 72])

        f2 = torch.cat([e3,c1],1)
        print("[Debug] f2.shape: ", f2.shape) # [Debug] f2.shape:  torch.Size([32, 128, 56, 72])

        e2 = self.expand2(f2)
        print("[Debug] e2.shape: ", e2.shape) # [Debug] e2.shape:  torch.Size([32, 32, 112, 144])

        f1 = torch.cat([e2,c0],1)
        print("[Debug] f1.shape: ", f1.shape) # [Debug] f1.shape:  torch.Size([32, 96, 112, 144])

        out = self.top(f1)
        print("[Debug] out.shape: ", out.shape) #

        import pdb; pdb.set_trace()

        return out