import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import learn2learn as l2l
from src.models.utils import convolution_block, transpose_convolution_block, \
    contract_block, bottom_block, expand_block, top_block
    
from src.dataset.dataloader import DataGenerator
from piqa import SSIM


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

        # # [DEBUG] -- Check bottom shape
        # print("[Debug] bb.shape: ", bb.shape) #[Debug] bb.shape:  torch.Size([1, 512, 28, 36])
        # print("[Debug] c4.shape: ", c4.shape) #[Debug] p4.shape:  torch.Size([1, 512, 14, 18])
        f4 = torch.cat([bb,c4],1)
        # print("[Debug] f4.shape: ", f4.shape) #[Debug] f4.shape:  torch.Size([1, 1024, 28, 36])

        e4 = self.expand4(f4)
        # print("[Debug] e4.shape: ", e4.shape) #[Debug] e4.shape:  torch.Size([1, 256, 56, 72])

        f3 = torch.cat([e4,c3],1)
        # print("[Debug] f3.shape: ", f3.shape) #

        e3 = self.expand3(f3)
        # print("[Debug] e3.shape: ", e3.shape) #

        f2 = torch.cat([e3,c2],1)
        # print("[Debug] f2.shape: ", f2.shape) #

        e2 = self.expand2(f2)
        # print("[Debug] e2.shape: ", e2.shape) #

        f1 = torch.cat([e2,c1],1)
        # print("[Debug] f1.shape: ", f1.shape) #

        out = self.top(f1)
        # print("[Debug] out.shape: ", out.shape) #
        out = torch.sigmoid(out)

        return out

    

def train_maml(cfg):    
    dg = DataGenerator(num_exposures=cfg.TRAIN.NUM_EXPOSURES)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr=0.005
    maml_lr=0.01
    
    model = UNet(in_size=3, out_size=3, num_filters=8).double()
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    loss_func = nn.MSELoss()
    
    # I believe this is num_inner_updates (?) in the cfg -- may want change it from 1 and just use that directly :P
    fas=3
    iterations = 100
    
    ssim = SSIM().double().cuda() if device == 'cuda' else SSIM().double()
    
    # Reference https://github.com/learnables/learn2learn/blob/master/examples/vision/meta_mnist.py
    for iteration in range(iterations):
        print("ITERATION #", iteration)
        iteration_error = 0.0
        # iteration_acc = 0.0
        iteration_ssim = 0
        
        train, test = dg.sample_batch('meta_train', cfg.TRAIN.BATCH_SIZE)
        train = torch.from_numpy(train).to(device)
        test = torch.from_numpy(test).to(device)
        
        for batch_index in range(cfg.TRAIN.BATCH_SIZE):
            print("Batch", batch_index)
            learner = meta_model.clone()
            
            # Separate data into adaptation/evalutation sets
            adaptation_data, adaptation_labels = train[0, batch_index, ...].permute(0, 3, 1, 2), train[1, batch_index, ...].permute(0, 3, 1, 2)
            evaluation_data, evaluation_labels = test[0, batch_index, ...].permute(0, 3, 1, 2), test[1, batch_index, ...].permute(0, 3, 1, 2)

            # If just calling a forward (i.e on adaptation data and don't want gradients to save space
            #, create a new func w decortor @torch.no_grad)            

            # Fast Adaptation
            for step in range(fas):
                train_error = loss_func(learner(adaptation_data), torch.clip(adaptation_labels, 0, 1))
                learner.adapt(train_error)
    
            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = loss_func(predictions, torch.clip(evaluation_labels, 0, 1))
            valid_error /= len(evaluation_data)
            
            # Will return avg ssim 
            valid_ssim = ssim(predictions, torch.clip(evaluation_labels, 0, 1)).item()
            
            iteration_error += valid_error
            iteration_ssim += valid_ssim
    
        iteration_error /= cfg.TRAIN.BATCH_SIZE
        # iteration_acc /= tps
        iteration_ssim /= cfg.TRAIN.BATCH_SIZE
        print('Train Loss : {:.3f} Train SSIM : {:.3f}'.format(iteration_error.item(), valid_ssim))

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()