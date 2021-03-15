import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import learn2learn as l2l
import matplotlib.pyplot as plt
from piqa import SSIM

from src.dataset.dataloader import DataGenerator
from src.core.utils import get_GPU_usage
from src.core.loss import ExpandNetLoss, HaarLoss, LPIPSLoss, SSIMLoss
from src.models.UNet import UNet
from src.dataset.hdr_visualization import visualize_hdr_image

def train_maml(cfg):    
    dg = DataGenerator(num_exposures=cfg.TRAIN.NUM_EXPOSURES)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr=0.005
    maml_lr=0.01
    
    model = UNet(in_size=3, out_size=3, num_filters=8).double()
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    # loss_func = nn.MSELoss()
    loss_func = ExpandNetLoss()
    
    # I believe this is num_inner_updates (?) in the cfg -- may want change it from 1 and just use that directly :P
    fas=3
    
    ssim = SSIM().double().cuda() if device == 'cuda' else SSIM().double()
    
    ssims = []
    losses = []
    
    # Reference https://github.com/learnables/learn2learn/blob/master/examples/vision/meta_mnist.py
    for iteration in range(cfg.TRAIN.NUM_META_TR_ITER):
        print("ITERATION #", iteration)
        iteration_error = 0.0
        # iteration_acc = 0.0
        iteration_ssim = 0
        
        train, test = dg.sample_batch('meta_train', cfg.TRAIN.BATCH_SIZE)
        train = torch.from_numpy(train).to(device)
        test = torch.from_numpy(test).to(device)
        
        for batch_index in range(cfg.TRAIN.BATCH_SIZE):
            get_GPU_usage(f'Index {batch_index}')
            print("Index", batch_index)
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
            
            # Plot the first batch index
            if not batch_index:
              fig, ax = plt.subplots(nrows=1,ncols=3)
              ax[0].imshow(visualize_hdr_image(predictions[0].detach().cpu().permute(1, 2, 0).numpy()))
              ax[0].axis('off')
              ax[0].set_title('Predicted (GCed)')
              ax[1].imshow(evaluation_data[0].detach().cpu().permute(1, 2, 0).numpy())
              ax[1].axis('off')
              ax[1].set_title('Original Exposure Shot')
              ax[2].imshow(visualize_hdr_image(torch.clip(adaptation_labels[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy()))
              ax[2].axis('off')
              ax[2].set_title('HDR (GCed)')
              fig.savefig(f'test{iteration}_{batch_index}.png', bbox_inches='tight')
              plt.close()
            
            # Will return avg ssim 
            valid_ssim = ssim(predictions, torch.clip(evaluation_labels, 0, 1)).item()
            
            iteration_error += valid_error
            iteration_ssim += valid_ssim

    
        iteration_error /= cfg.TRAIN.BATCH_SIZE
        # iteration_acc /= tps
        iteration_ssim /= cfg.TRAIN.BATCH_SIZE
        print('Train Loss : {:.3f} Train SSIM : {:.3f}'.format(iteration_error.item(), valid_ssim))
        
        ssims.append(iteration_ssim)
        losses.append(iteration_error)

        # Take the meta-learning step
        opt.zero_grad()
        iteration_error.backward()
        opt.step()
        plt.figure()
        
    # Plot losses and ssims
    plt.plot(np.arange(1, len(losses)+1), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss Across Iterations")
    plt.savefig('loss_iterations.png')

    plt.figure()
    plt.plot(np.arange(1, len(ssims)+1), ssims)
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.title("SSIM Across Iterations")
    plt.savefig('ssim_iterations.png')
