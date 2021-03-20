import torch
import argparse
from os import path as osp
import os
from tqdm import tqdm
import numpy as np
import learn2learn as l2l
from sklearn.model_selection import train_test_split
import skimage.io as io
import logging

from src.core.config import update_cfg, get_cfg_defaults
from src.models.UNet import UNet
from src.models.metaHDR import evaluate_maml, evaluate_single_maml
from src.dataset.dataloader import DataGenerator
from src.core.loss import get_loss_func
from src.core.utils import create_logger

def main(args):
    print("--- Evaluating on meta-test set ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source_directory = args.model_dir
    use_last_flag = args.use_last

    # Make evaluation folder for test images
    evaluation_figure_output_dir = osp.join(source_directory, 'evaluation_output')
    os.makedirs(evaluation_figure_output_dir, exist_ok=True)

    # Make the subfolders
    single_dir = os.path.join(evaluation_figure_output_dir, "single")
    adapt_debevec_dir = os.path.join(evaluation_figure_output_dir, "adapt_debevec")
    adapt_hdrcnn_dir = os.path.join(evaluation_figure_output_dir, "adapt_hdrcnn")

    os.makedirs(single_dir, exist_ok=True)
    os.makedirs(adapt_debevec_dir, exist_ok=True)
    os.makedirs(adapt_hdrcnn_dir, exist_ok=True)
    os.makedirs(evaluation_figure_output_dir, exist_ok=True)

    # Grab config
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    # Make sure loss_func from config is valid, then get it
    assert cfg.EVAL.LOSS_FUNC in ['ExpandNetLoss', 'HaarLoss', 'LPIPSLoss', 'LPIPSLoss_L1', 'SSIMLoss'], f"[CONFIG] evaluation loss function '{cfg.EVAL.LOSS_FUNC}' not valid"
    loss_func = get_loss_func(cfg.EVAL.LOSS_FUNC)

    # Grad test data -- all of it
    dg = DataGenerator(num_exposures=cfg.EVAL.NUM_EXPOSURES)
    all_test_data = dg.meta_test_data

    # Create logger
    logger = create_logger(evaluation_figure_output_dir, phase='eval', level=logging.CRITICAL)
    logger.critical(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.critical(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    # Grab model checkpoint
    if use_last_flag:
        model_path = osp.join(source_directory, 'model_last.pth.tar')
    else:
        model_path = osp.join(source_directory, 'model_best.pth.tar')

    checkpoint = torch.load(model_path)
    best_performance = checkpoint['performance']
    best_epoch = checkpoint['epoch']
    logger.critical(f"During training: Best Epoch: {best_epoch}, Best SSIM: {best_performance}")

    # Define blank model to load weights into
    model = UNet(in_size=3, out_size=3, num_filters=8).double().to(device)
    meta_model = l2l.algorithms.MAML(model, lr=cfg.EVAL.TASK_LR)
    logger.critical(f"Loading pre-trained model from --> {model_path}")
    meta_model.load_state_dict(checkpoint['unet_state_dict'])
    logger.critical(f"Successfully loaded pre-trained model from --> {model_path}")

    # Perform single-shot evaluation
    logger.critical("Performing Single-Shot Evaluation")
    eval_single_ssim = 0.0
    eval_single_psnr = 0.0
    idx = 0
    for i in tqdm(range(all_test_data.shape[0])):
        for j in range(1, all_test_data.shape[1]):
            input_test_image = all_test_data[np.newaxis, i, j]
            input_test_label = all_test_data[np.newaxis, i, 0]
            _, test_ssim, test_psnr = evaluate_single_maml(meta_model, loss_func, input_test_image, input_test_label, idx, device=device, visualize_flag=True, visualize_dir=single_dir)

            eval_single_ssim+=test_ssim
            eval_single_psnr+=test_psnr
            idx += 1
    eval_single_ssim /= (all_test_data.shape[0]*(all_test_data.shape[1]-1))
    eval_single_psnr /= (all_test_data.shape[0]*(all_test_data.shape[1]-1))

    # Perform adaptive evaluation
    logger.critical("Performing Adaptive Evaluation using Debevec labels")
    eval_adaptive_ssim = 0.0
    eval_adaptive_psnr = 0.0
    idx = 0
    for curr_idx in tqdm(range(all_test_data.shape[0])):
        cur_batch = all_test_data[np.newaxis, curr_idx]
        tr_images, ts_images = [], []
        tr_labels, ts_labels = [], []
        for image_set in cur_batch:
            # Train and Test for each set of exposures
            tr, ts = train_test_split(np.arange(1, cfg.EVAL.NUM_EXPOSURES+1), test_size=1)
            
            cur_tr_images, cur_tr_labels = [], []
            for i in tr:
                cur_tr_images.append(image_set[i, ...])
                cur_tr_labels.append(image_set[0, ...])
            tr_images.append(np.stack(cur_tr_images))
            tr_labels.append(np.stack(cur_tr_labels))
            
            cur_ts_images, cur_ts_labels = [], []
            for i in ts:
                cur_ts_images.append(image_set[i, ...])
                cur_ts_labels.append(image_set[0, ...])
            ts_images.append(np.stack(cur_ts_images))
            ts_labels.append(np.stack(cur_ts_labels))
        
        tr_images = np.stack(tr_images)
        tr_labels = np.stack(tr_labels)
        ts_images = np.stack(ts_images)
        ts_labels = np.stack(ts_labels)
        
        eval_train = np.stack([tr_images, tr_labels])
        eval_test = np.stack([ts_images, ts_labels])

        import pdb; pdb.set_trace()
        _, test_ssim, test_psnr = evaluate_maml(meta_model, loss_func, eval_train, eval_test, idx, cfg.EVAL.NUM_TASK_TR_ITER, device=device, model_type=cfg.TRAIN.MODEL, visualize_flag=True, visualize_dir=adapt_debevec_dir)
        idx += 1
        
        eval_adaptive_ssim += test_ssim
        eval_adaptive_psnr += test_psnr

    eval_adaptive_ssim /= all_test_data.shape[0]
    eval_adaptive_psnr /= all_test_data.shape[0]

    # Perform adaptive evaluation
    logger.critical("Performing Adaptive Evaluation using HDRCNN labels")
    eval_adaptive_hdrcnn_ssim = 0.0
    eval_adaptive_hdrcnn_psnr = 0.0
    # cur_indices = [1,   4,  11,  20,  23,  27,  36,  45,  48,  \
    #                 53,  56,  65,  70, 73,  82,  84, 126, 137, \
    #                 138, 150, 162, 171, 173, 175, 194, 195, 209, \
    #                 221, 224, 230, 275, 288, 294, 306, 342, 360, \
    #                 375, 394, 397, 405, 430, 438, 439, 448, 450]
    from pathlib import Path
    load_dir = Path(__file__).parent/'data/LDR-HDR-pair_Dataset/TestOutputs' # /home/users/edwinpan/MetaHDR/data/LDR-HDR-pair_Dataset/TestOutputs
    idx = 0 
    for curr_idx in tqdm(range(all_test_data.shape[0])):
        cur_batch = all_test_data[np.newaxis, curr_idx]
        tr_images, ts_images = [], []
        tr_labels, ts_labels = [], []
        for image_set in cur_batch:
            # Train and Test for each set of exposures
            tr, ts = train_test_split(np.arange(1, cfg.EVAL.NUM_EXPOSURES+1), test_size=1)
            cur_tr_images, cur_tr_labels = [], []
            for i in tr:
                image_idx = curr_idx + 1
                cur_tr_images.append(image_set[i, ...])
                if i == 1: # exposure p2
                    image_idx += all_test_data.shape[0]
                elif i == 2: #exposure 0
                    pass
                elif i == 3: # expusre n2
                    image_idx += all_test_data.shape[0] * 2
                cur_label = io.imread(load_dir/f'{image_idx:06d}_out.png').astype(np.float64) / 255
                cur_tr_labels.append(cur_label)
            tr_images.append(np.stack(cur_tr_images))
            tr_labels.append(np.stack(cur_tr_labels))
            cur_ts_images, cur_ts_labels = [], []
            for i in ts:
                cur_ts_images.append(image_set[i, ...])
                cur_ts_labels.append(image_set[0, ...])
            ts_images.append(np.stack(cur_ts_images))
            ts_labels.append(np.stack(cur_ts_labels))
        
        tr_images = np.stack(tr_images)
        tr_labels = np.stack(tr_labels)
        ts_images = np.stack(ts_images)
        ts_labels = np.stack(ts_labels)
        
        eval_train = np.stack([tr_images, tr_labels])
        eval_test = np.stack([ts_images, ts_labels])

        # import pdb; pdb.set_trace()
        _, test_ssim, test_psnr = evaluate_maml(meta_model, loss_func, eval_train, eval_test, idx, cfg.EVAL.NUM_TASK_TR_ITER, device=device, model_type=cfg.TRAIN.MODEL, visualize_flag=True, visualize_dir=adapt_hdrcnn_dir)
        idx += 1
        
        eval_adaptive_hdrcnn_ssim += test_ssim
        eval_adaptive_hdrcnn_psnr += test_psnr

    eval_adaptive_hdrcnn_ssim /= all_test_data.shape[0]
    eval_adaptive_hdrcnn_psnr /= all_test_data.shape[0]

    logger.critical("[Evaluation Results] Average Single-Shot Evaluation SSIM : {:.3f}".format(eval_single_ssim))
    logger.critical("[Evaluation Results] Average Single-Shot Evaluation PSNR : {:.3f}".format(eval_single_psnr))
    logger.critical("[Evaluation Results] Average Debevec Adapted Evaluation SSIM : {:.3f}".format(eval_adaptive_ssim))
    logger.critical("[Evaluation Results] Average Debevec Adapted Evaluation PSNR : {:.3f}".format(eval_adaptive_psnr))
    logger.critical("[Evaluation Results] Average HDRCNN Adapted Evaluation SSIM : {:.3f}".format(eval_adaptive_hdrcnn_ssim))
    logger.critical("[Evaluation Results] Average HDRCNN Adapted Evaluation PSNR : {:.3f}".format(eval_adaptive_hdrcnn_psnr))

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Path to directory with outputs from MetaHDR training cycle.')
    parser.add_argument('--cfg', type=str, help='cfg file path')
    parser.add_argument('--use_last', dest='use_last', action='store_true', help='Use to evaluate the last model in training iters. Default is to evaluate the best model from training.')

    args = parser.parse_args()

    main(args)