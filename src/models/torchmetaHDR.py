import numpy as np
import torch
import logging
from torch import nn, optim
from torch.nn import functional as F
import learn2learn as l2l
import matplotlib.pyplot as plt
from piqa import SSIM

from src.dataset.dataloader import DataGenerator
from src.dataset.hdr_visualization import visualize_hdr_image
from src.core.utils import get_GPU_usage
from src.core.loss import ExpandNetLoss, HaarLoss, LPIPSLoss, SSIMLoss
from src.models.UNet import UNet
from src.models.utils import save_model

logger = logging.getLogger(__name__)

@torch.no_grad
def eval_maml(learner, loss_func, train, test, batch_size, num_inner_updates, curr_meta_iter, ssim=None, device=None, log_dir=None):
    model = learner.clone()
    test_error, test_ssim = 0, 0
    for batch_idx in range(batch_size):
        # adaptation_data, adaptation_labels = train[0, batch_idx, ...].permute(0, 3, 1, 2), train[1, batch_idx, ...].permute(0, 3, 1, 2)
        evaluation_data, evaluation_labels = test[0, batch_idx, ...].permute(0, 3, 1, 2), test[1, batch_idx, ...].permute(0, 3, 1, 2)

        # for _ in range(num_inner_updates):
        #     train_error = loss_func(model(adaptation_data), torch.clip(adaptation_labels, 0, 1))
        #     model.adapt(train_error)

        test_predictions = model(evaluation_data)
        test_error += loss_func(test_predictions, torch.clip(evaluation_labels, 0, 1))/len(test_predictions)
        test_ssim += ssim(test_predictions, torch.clip(evaluation_labels, 0, 1)).item()
    
    test_error /= batch_size
    test_ssim /= batch_size

    # print('[Meta-Validation {}] Validation Loss : {:.3f} Validation SSIM : {:.3f}'.format(curr_meta_iter, test_error.item(), test_ssim))
    logger.info('[Meta-Validation {}] Validation Loss : {:.3f} Validation SSIM : {:.3f}'.format(curr_meta_iter, test_error.item(), test_ssim))

    fig, ax = plt.subplots(nrows=1,ncols=3)
    ax[0].imshow(visualize_hdr_image(test_predictions[0].detach().cpu().permute(1, 2, 0).numpy()))
    ax[0].axis('off')
    ax[0].set_title('Predicted (GCed)')
    ax[1].imshow(evaluation_data[0].detach().cpu().permute(1, 2, 0).numpy())
    ax[1].axis('off')
    ax[1].set_title('Original Exposure Shot')
    ax[2].imshow(visualize_hdr_image(torch.clip(adaptation_labels[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy()))
    ax[2].axis('off')
    ax[2].set_title('HDR')
    fig.savefig(f'{log_dir}/meta_val_{curr_meta_iter}.png', bbox_inches='tight')
    plt.close()

    return test_error, test_ssim


def train_maml(cfg, log_dir):    
    dg = DataGenerator(num_exposures=cfg.TRAIN.NUM_EXPOSURES)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr=cfg.TRAIN.META_LR
    maml_lr=cfg.TRAIN.TASK_LR
    
    model = UNet(in_size=3, out_size=3, num_filters=8).double()
    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)
    # loss_func = nn.MSELoss()
    loss_func = ExpandNetLoss()
    
    ssim = SSIM().double().cuda() if device == 'cuda' else SSIM().double()
    
    pre_ssims = []
    ssims = []
    losses = []
    
    best_performance = 0.0 # for tracking model progress
    # Reference https://github.com/learnables/learn2learn/blob/master/examples/vision/meta_mnist.py
    for iteration in range(cfg.TRAIN.NUM_META_TR_ITER):
        # print("ITERATION #", iteration)
        iteration_error = 0.0
        iteration_ssim = 0
        
        train, test = dg.sample_batch('meta_train', cfg.TRAIN.BATCH_SIZE)
        train = torch.from_numpy(train).to(device)
        test = torch.from_numpy(test).to(device)

        for batch_index in range(cfg.TRAIN.BATCH_SIZE):
            get_GPU_usage(f'Index {batch_index}')
            # print("Index", batch_index)
            learner = meta_model.clone()
            get_GPU_usage(f'post clone {batch_index}')

            # Separate data into adaptation/evalutation sets
            adaptation_data, adaptation_labels = train[0, batch_index, ...].permute(0, 3, 1, 2), train[1, batch_index, ...].permute(0, 3, 1, 2)
            evaluation_data, evaluation_labels = test[0, batch_index, ...].permute(0, 3, 1, 2), test[1, batch_index, ...].permute(0, 3, 1, 2)
            get_GPU_usage(f'post data split {batch_index}')

            # If just calling a forward (i.e on adaptation data and don't want gradients to save space
            #, create a new func w decortor @torch.no_grad)            

            # Fast Adaptation -- first iter
            if not batch_index:
                first_train_pred = learner(adaptation_data)
                train_error = loss_func(first_train_pred, torch.clip(adaptation_labels, 0, 1))
                learner.adapt(train_error)
                pre_train_ssim = ssim(first_train_pred, torch.clip(adaptation_labels, 0, 1)).item()
                # print('[Pre-Train {}] Train Loss : {:.3f} Train SSIM : {:.3f}'.format(iteration, train_error.item(), pre_train_ssim))
                logger.info('[Pre-Train {}] Train Loss : {:.3f} Train SSIM : {:.3f}'.format(iteration, train_error.item(), pre_train_ssim))

                # Fast Adaptation -- rest of the iters
                for step in range(cfg.TRAIN.NUM_TASK_TR_ITER-1):
                    train_error = loss_func(learner(adaptation_data), torch.clip(adaptation_labels, 0, 1))
                    learner.adapt(train_error)
            else:
                # Fast Adaptation -- all iters
                for step in range(cfg.TRAIN.NUM_TASK_TR_ITER):
                    train_error = loss_func(learner(adaptation_data), torch.clip(adaptation_labels, 0, 1))
                    learner.adapt(train_error)
            get_GPU_usage(f'post fast adapt {batch_index}')

            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = loss_func(predictions, torch.clip(evaluation_labels, 0, 1))
            valid_error /= len(evaluation_data)
            
            # Plot the first batch index
            if batch_index == cfg.TRAIN.BATCH_SIZE-1:
              fig, ax = plt.subplots(nrows=1,ncols=3)
              ax[0].imshow(visualize_hdr_image(predictions[0].detach().cpu().permute(1, 2, 0).numpy()))
              ax[0].axis('off')
              ax[0].set_title('Predicted (GCed)')
              ax[1].imshow(evaluation_data[0].detach().cpu().permute(1, 2, 0).numpy())
              ax[1].axis('off')
              ax[1].set_title('Original Exposure Shot')
              ax[2].imshow(visualize_hdr_image(torch.clip(adaptation_labels[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy()))
              ax[2].axis('off')
              ax[2].set_title('HDR')
              fig.savefig(f'{log_dir}/test{iteration}.png', bbox_inches='tight')
              plt.close()
            
            # Will return avg ssim 
            valid_ssim = ssim(predictions, torch.clip(evaluation_labels, 0, 1)).item()
            
            iteration_error += valid_error
            iteration_ssim += valid_ssim

    
        iteration_error /= cfg.TRAIN.BATCH_SIZE
        iteration_ssim /= cfg.TRAIN.BATCH_SIZE
        # print('[Post-Train {}] Train Loss : {:.3f} Train SSIM : {:.3f}'.format(iteration, iteration_error.item(), valid_ssim))
        logger.info('[Post-Train {}] Train Loss : {:.3f} Train SSIM : {:.3f}'.format(iteration, iteration_error.item(), valid_ssim))

        ssims.append(iteration_ssim)
        losses.append(iteration_error.item())

        # Meta-validation
        # if (iteration!=0) and iteration % cfg.TEST_PRINT_INTERVAL == 0:
        if iteration==0:
            val_train, val_test = dg.sample_batch('meta_val', 1)
            # val_train = torch.from_numpy(val_train).to(device)
            val_test = torch.from_numpy(val_test).to(device)

            _, meta_val_ssim = eval_maml(learner, loss_func, val_train, val_test, cfg.TRAIN.BATCH_SIZE, cfg.TRAIN.NUM_TASK_TR_ITER, iteration, ssim=ssim, device=device, log_dir=log_dir)

            if meta_val_ssim > best_performance:
                logger.info('Best performance achieved, saving it!')
                save_model(learner, iteration, meta_val_ssim, log_dir)
                best_performance = meta_val_ssim

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
    plt.savefig(f'{log_dir}/loss_iterations.png')

    plt.figure()
    plt.plot(np.arange(1, len(ssims)+1), ssims)
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.title("SSIM Across Iterations")
    plt.savefig(f'{log_dir}/ssim_iterations.png')
