import numpy as np
import torch
import os
import logging
from torch import nn, optim
from torch.nn import functional as F
import learn2learn as l2l
import matplotlib.pyplot as plt
from piqa import SSIM, PSNR
from functools import partial
from progress.bar import Bar

from src.dataset.dataloader import DataGenerator
from src.dataset.dataloader_patch import PatchHDRDataset, _load_pkl
from src.dataset.hdr_visualization import visualize_hdr_image
from src.core.utils import get_GPU_usage
from src.core.loss import get_loss_func
from src.models.UNet import UNet, Resnet
from src.models.utils import save_best_model, save_last_model, count_parameters

logger = logging.getLogger(__name__)

@torch.no_grad()
def evaluate_single_maml(model, loss_func, image, label, idx, device=None, visualize_flag=False, visualize_dir=None):
    """
    Evaluate 1 test image using meta-params as input
    """
    # Cast as torch tensor & send data to device
    input_image = torch.from_numpy(image).to(device).permute(0, 3, 1, 2)
    input_label = torch.from_numpy(label).to(device).permute(0, 3, 1, 2)

    # Instantiate evaluation metric (ssim)
    ssim = SSIM().double().cuda() if device == 'cuda' else SSIM().double()
    psnr = PSNR().double().cuda() if device == 'cuda' else PSNR().double()

    # Pass through model
    test_prediction = model(input_image)

    test_error = loss_func(test_prediction, torch.clip(input_label, 0, 1))
    test_ssim = ssim(test_prediction, torch.clip(input_label, 0, 1)).item()
    test_psnr = psnr(test_prediction, torch.clip(input_label, 0, 1)).item()

    logger.critical(f"[Single-Shot {idx:03d}] SSIM: {test_ssim}, PSNR: {test_psnr}")

    if visualize_flag:
        fig, ax = plt.subplots(nrows=1,ncols=3)
        ax[1].imshow(visualize_hdr_image(test_prediction[0].detach().cpu().permute(1, 2, 0).numpy()))
        ax[1].axis('off')
        ax[1].set_title('Predicted HDR')
        ax[0].imshow(input_image[0].detach().cpu().permute(1, 2, 0).numpy())
        ax[0].axis('off')
        ax[0].set_title('Original LDR')
        ax[2].imshow(visualize_hdr_image(torch.clip(input_label[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy()))
        ax[2].axis('off')
        ax[2].set_title('True HDR')
        fig.savefig(f'{visualize_dir}/evaluation_single{idx:03d}.png', bbox_inches='tight')
        plt.close()

    return test_prediction[0].detach().cpu().permute(1, 2, 0).numpy(), test_ssim, test_psnr


def evaluate_maml(model, loss_func, train, test, idx, num_inner_updates, device=None, model_type=None, visualize_flag=False, visualize_dir=None):
    """
    Evaluate 1 test task using task-specific adaptation.
    """
    # Cast as torch tensor & send data to device
    train = torch.from_numpy(train).to(device)
    test = torch.from_numpy(test).to(device)

    # Instantiate evaluation metric (ssim)
    ssim = SSIM().double().cuda() if device == 'cuda' else SSIM().double()
    psnr = PSNR().double().cuda() if device == 'cuda' else PSNR().double()

    # Pass each batch through
    test_error, test_ssim, test_psnr = 0.0, 0.0, 0.0
    for batch_index in range(1):
        learner = model.clone()
        adaptation_data, adaptation_labels = train[0, batch_index, ...].permute(0, 3, 1, 2), train[1, batch_index, ...].permute(0, 3, 1, 2)
        evaluation_data, evaluation_labels = test[0, batch_index, ...].permute(0, 3, 1, 2), test[1, batch_index, ...].permute(0, 3, 1, 2)

        for _ in range(num_inner_updates):
            train_error = loss_func(learner(adaptation_data), torch.clip(adaptation_labels, 0, 1))
            if model_type == 'Resnet':
                learner.adapt(train_error, allow_nograd=True, allow_unused=True)
            else:
                learner.adapt(train_error)

        test_predictions = learner(evaluation_data)
        test_error += loss_func(test_predictions, torch.clip(evaluation_labels, 0, 1))/len(test_predictions)
        test_ssim += ssim(test_predictions, torch.clip(evaluation_labels, 0, 1)).item()
        test_psnr += psnr(test_predictions, torch.clip(evaluation_labels, 0, 1)).item()

        logger.critical(f"[{os.path.basename(os.path.normpath(visualize_dir))} {idx:03d}] SSIM: {test_ssim}, PSNR: {test_psnr}")

        if visualize_flag:
            fig, ax = plt.subplots(nrows=1,ncols=3)
            ax[1].imshow(visualize_hdr_image(test_predictions[0].detach().cpu().permute(1, 2, 0).numpy()))
            ax[1].axis('off')
            ax[1].set_title('Predicted HDR')
            ax[0].imshow(evaluation_data[0].detach().cpu().permute(1, 2, 0).numpy())
            ax[0].axis('off')
            ax[0].set_title('Original LDR')
            ax[2].imshow(visualize_hdr_image(torch.clip(evaluation_labels[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy()))
            ax[2].axis('off')
            ax[2].set_title('True HDR')
            fig.savefig(f'{visualize_dir}/evaluation_adapt{idx:03d}.png', bbox_inches='tight')
            plt.close()

    return test_predictions[0].detach().cpu().permute(1, 2, 0).numpy(), test_ssim, test_psnr

@torch.no_grad()
def validate_maml(learner, loss_func, val_dl, val_dl_iter, batch_size, num_inner_updates, curr_meta_iter, ssim=None, device=None, log_dir=None):
    model = learner.clone()


    test_error, test_ssim = 0, 0
    for batch_idx in range(batch_size):

        try:
            train, test = next(val_dl_iter)
        except StopIteration:
            val_dl_iter = iter(val_dl)
            train, test = next(val_dl_iter)

        test = torch.from_numpy(test).to(device)

        adaptation_data, adaptation_labels = train[0, ...], train[1, ...]
        evaluation_data, evaluation_labels = test[0, ...].permute(0, 3, 1, 2), test[1, ...].permute(0, 3, 1, 2)

        # TODO: Wrap all these functions in 1 Trainer class
        # TODO: Enable adaptation for validation
        test_predictions = model(evaluation_data)
        test_error += loss_func(test_predictions, torch.clip(evaluation_labels, 0, 1))/len(test_predictions)
        test_ssim += ssim(test_predictions, torch.clip(evaluation_labels, 0, 1)).item()
    
    test_error /= batch_size
    test_ssim /= batch_size

    logger.info('[Meta-Validation {}] Validation Loss : {:.3f} Validation SSIM : {:.3f}'.format(curr_meta_iter, test_error.item(), test_ssim))

    fig, ax = plt.subplots(nrows=1,ncols=3)
    ax[1].imshow(visualize_hdr_image(test_predictions[0].detach().cpu().permute(1, 2, 0).numpy()))
    ax[1].axis('off')
    ax[1].set_title('Predicted HDR')
    ax[0].imshow(evaluation_data[0].detach().cpu().permute(1, 2, 0).numpy())
    ax[0].axis('off')
    ax[0].set_title('Original LDR')
    ax[2].imshow(visualize_hdr_image(np.clip(adaptation_labels[0], 0, 1)))
    ax[2].axis('off')
    ax[2].set_title('True HDR')
    fig.savefig(f'{log_dir}/meta_val_{curr_meta_iter:03d}.png', bbox_inches='tight')
    plt.close()

    return test_error, test_ssim


def train_maml(cfg, log_dir):    
    # dg = DataGenerator(num_exposures=cfg.TRAIN.NUM_EXPOSURES, include_unet_outputs=cfg.TRAIN.INCLUDE_UNET_OUTPUTS)
    i_dataset_train_posfix_list = _load_pkl('/home/users/edwinpan/MetaHDR/i_dataset_train.pkl')
    i_dataset_test_posfix_list = _load_pkl('/home/users/edwinpan/MetaHDR/i_dataset_test.pkl')
    hdr_prefix = '/scratch/users/edwinpan/data/SingleHDR_training_data/'
    
    dl = PatchHDRDataset(hdr_prefix, i_dataset_train_posfix_list, n_way=cfg.TRAIN.NUM_EXPOSURES, is_training=True)
    val_dl = PatchHDRDataset(hdr_prefix, i_dataset_test_posfix_list, n_way=cfg.TRAIN.NUM_EXPOSURES, is_training=False)
    dl_iter = iter(dl)
    val_dl_iter = iter(val_dl)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr=cfg.TRAIN.META_LR
    maml_lr=cfg.TRAIN.TASK_LR
    
    if cfg.TRAIN.MODEL == 'Unet':
        logger.info("Using Unet model for MAML inner")
        model = UNet(in_size=3, out_size=3, num_filters=8).double()
    elif cfg.TRAIN.MODEL == 'Resnet':
        logger.info("Using Unet model w/ Resnet contract for MAML inner")
        model = Resnet(in_size=3, out_size=3).double()

    model.to(device)
    meta_model = l2l.algorithms.MAML(model, lr=maml_lr)
    opt = optim.Adam(meta_model.parameters(), lr=lr)

    loss_func = get_loss_func(cfg.TRAIN.LOSS_FUNC)
    
    ssim = SSIM().double().cuda() if device == 'cuda' else SSIM().double()
    
    pre_ssims = []
    pre_loss = []
    ssims = []
    losses = []
    
    best_performance = 0.0 # for tracking model progress
    for iteration in range(cfg.TRAIN.NUM_META_TR_ITER):
        iteration_error = 0.0
        iteration_ssim = 0
    
        summary_string = ''
        bar = Bar(f'[Train] Epoch {iteration + 1}/{cfg.TRAIN.NUM_META_TR_ITER}', fill='#', max=cfg.TRAIN.BATCH_SIZE)

        for batch_index in range(cfg.TRAIN.BATCH_SIZE):
            learner = meta_model.clone()

            try:
                train, test = next(dl_iter)
            except StopIteration:
                dl_iter = iter(dl)
                train, test = next(dl_iter)
            train = torch.from_numpy(train).to(device)
            test = torch.from_numpy(test).to(device)

            # Separate data into adaptation/evalutation sets
            adaptation_data, adaptation_labels = train[0, ...].permute(0, 3, 1, 2), train[1, ...].permute(0, 3, 1, 2)
            evaluation_data, evaluation_labels = test[0, ...].permute(0, 3, 1, 2), test[1, ...].permute(0, 3, 1, 2)          

            # Fast Adaptation -- first iter
            if not batch_index:
                first_train_pred = learner(adaptation_data)
                train_error = loss_func(first_train_pred, torch.clip(adaptation_labels, 0, 1))
                if cfg.TRAIN.MODEL == 'Resnet':
                    learner.adapt(train_error, allow_nograd=True, allow_unused=True)
                else:
                    learner.adapt(train_error)

                pre_train_ssim = ssim(first_train_pred, torch.clip(adaptation_labels, 0, 1)).item()
                pre_ssims.append(pre_train_ssim)
                pre_loss.append(train_error.item())

                # Fast Adaptation -- rest of the iters
                for step in range(cfg.TRAIN.NUM_TASK_TR_ITER-1):
                    train_error = loss_func(learner(adaptation_data), torch.clip(adaptation_labels, 0, 1))
                    if cfg.TRAIN.MODEL == 'Resnet':
                        learner.adapt(train_error, allow_nograd=True, allow_unused=True)
                    else:
                        learner.adapt(train_error)
            else:
                # Fast Adaptation -- all iters
                for step in range(cfg.TRAIN.NUM_TASK_TR_ITER):
                    train_error = loss_func(learner(adaptation_data), torch.clip(adaptation_labels, 0, 1))
                    if cfg.TRAIN.MODEL == 'Resnet':
                        learner.adapt(train_error, allow_nograd=True, allow_unused=True)
                    else:
                        learner.adapt(train_error)

            summary_string = f'({batch_index + 1}/{cfg.TRAIN.BATCH_SIZE}) | Total: {bar.elapsed_td} | ' \
                             f'ETA: {bar.eta_td:} | [pre] Loss: {pre_loss[-1]:.4f} | [pre] SSIM: {pre_ssims[-1]:.4f}'

            # Compute validation loss
            predictions = learner(evaluation_data)
            valid_error = loss_func(predictions, torch.clip(evaluation_labels, 0, 1))
            valid_error /= len(evaluation_data)
            
            if batch_index == cfg.TRAIN.BATCH_SIZE-1:
                # Record train summary string
                summary_string += f' | [post] Loss: {iteration_error.item()/(batch_index+1):.4f} | [post] SSIM: {valid_ssim:.4f}'
                logger.debug(summary_string)
                
                # Plot the last batch index
                fig, ax = plt.subplots(nrows=1,ncols=3)
                ax[0].imshow(visualize_hdr_image(predictions[0].detach().cpu().permute(1, 2, 0).numpy()))
                ax[0].axis('off')
                ax[0].set_title('Predicted')
                ax[1].imshow(evaluation_data[0].detach().cpu().permute(1, 2, 0).numpy())
                ax[1].axis('off')
                ax[1].set_title('Original Exposure Shot')
                ax[2].imshow(visualize_hdr_image(torch.clip(adaptation_labels[0], 0, 1).detach().cpu().permute(1, 2, 0).numpy()))
                ax[2].axis('off')
                ax[2].set_title('HDR')
                fig.savefig(f'{log_dir}/test{iteration:03d}.png', bbox_inches='tight')
                plt.close()
            
            # Will return avg ssim 
            valid_ssim = ssim(predictions, torch.clip(evaluation_labels, 0, 1)).item()
            
            iteration_error += valid_error
            iteration_ssim += valid_ssim

            import pdb; pdb.set_trace()


            bar.suffix = summary_string
            bar.next()

        bar.finish()

        iteration_error /= cfg.TRAIN.BATCH_SIZE
        iteration_ssim /= cfg.TRAIN.BATCH_SIZE

        # logger.info('[Post-Train {}] Train Loss : {:.3f} Train SSIM : {:.3f}'.format(iteration, iteration_error.item(), valid_ssim))

        ssims.append(iteration_ssim)
        losses.append(iteration_error.item())

        # Meta-validation
        if (iteration!=0) and iteration % cfg.TEST_PRINT_INTERVAL == 0:
            print("[Running Meta-Validation]")
            # val_train, val_test = dg.sample_batch('meta_val', cfg.TRAIN.VAL_BATCH_SIZE)

            _, meta_val_ssim = validate_maml(learner, loss_func, val_dl, val_dl_iter, cfg.TRAIN.VAL_BATCH_SIZE, cfg.TRAIN.NUM_TASK_TR_ITER, iteration, ssim=ssim, device=device, log_dir=log_dir)

            if meta_val_ssim > best_performance:
                logger.info('Best performance achieved, saving it!')
                save_best_model(learner, iteration, meta_val_ssim, log_dir)
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
    plt.plot(np.arange(1, len(pre_ssims)+1), pre_ssims)
    plt.xlabel("Iteration")
    plt.ylabel("SSIM")
    plt.legend(['post-ssim', 'pre-ssim'])
    plt.title("SSIM Across Iterations")
    plt.savefig(f'{log_dir}/ssim_iterations.png')

    logger.info('Saving last model for reference.')
    save_last_model(learner, cfg.TRAIN.NUM_META_TR_ITER, ssims[-1], log_dir)