import torch
import argparse
from os import path as osp
import os
from tqdm import tqdm
import numpy as np
import learn2learn as l2l
from sklearn.model_selection import train_test_split

from src.core.config import update_cfg, get_cfg_defaults
from src.models.UNet import UNet
from src.models.metaHDR import evaluate_maml, evaluate_single_maml
from src.dataset.dataloader import DataGenerator
from src.core.loss import get_loss_func

def main(args):
    print("--- Evaluating on meta-test set ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source_directory = args.model_dir
    use_best_flag = args.use_best

    # Make evaluation folder for test images
    evaluation_figure_output_dir = osp.join(source_directory, 'evaluation_output')
    os.makedirs(evaluation_figure_output_dir, exist_ok=True)

    # Grab config
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    # Make sure loss_func from config is valid, then get it
    assert cfg.EVAL.LOSS_FUNC in ['ExpandNetLoss', 'HaarLoss', 'LPIPSLoss', 'LPIPSLoss_L2', 'SSIMLoss'], f"[CONFIG] evaluation loss function '{cfg.EVAL.LOSS_FUNC}' not valid"
    loss_func = get_loss_func(cfg.EVAL.LOSS_FUNC)

    # Grab model checkpoint
    if use_best_flag:
        model_path = osp.join(source_directory, 'model_best.pth.tar')
    else:
        model_path = osp.join(source_directory, 'model_last.pth.tar')
    
    checkpoint = torch.load(model_path)
    best_performance = checkpoint['performance']
    best_epoch = checkpoint['epoch']
    print(f"During training: Best Epoch: {best_epoch}, Best SSIM: {best_performance}")

    # Define blank model to load weights into
    model = UNet(in_size=3, out_size=3, num_filters=8).double().to(device)
    meta_model = l2l.algorithms.MAML(model, lr=cfg.EVAL.TASK_LR)
    print(f"Loading pre-trained model from --> {model_path}")
    meta_model.load_state_dict(checkpoint['unet_state_dict'])
    print(f"Successfully loaded pre-trained model from --> {model_path}")

    # Grad test data -- all of it
    dg = DataGenerator(num_exposures=cfg.EVAL.NUM_EXPOSURES)

    all_test_data = dg.meta_test_data

    # # Perform single-shot evaluation
    # print("Performing Single-Shot Evaluation")
    # eval_single_ssim = 0.0
    # idx = 0
    # for i in tqdm(range(all_test_data.shape[0])):
    #     for j in range(1, all_test_data.shape[1]):
    #         input_test_image = all_test_data[np.newaxis, i, j]
    #         input_test_label = all_test_data[np.newaxis, i, 0]
    #         _, test_ssim = evaluate_single_maml(meta_model, loss_func, input_test_image, input_test_label, idx, device=device, visualize_flag=True, visualize_dir=evaluation_figure_output_dir)

    #         eval_single_ssim+=test_ssim
    #         idx += 1
    # eval_single_ssim /= (all_test_data.shape[0]*(all_test_data.shape[1]-1))
    # print("[Evaluation Results] Average Single-Shot Evaluation SSIM : {:.3f}".format(eval_single_ssim))

    # Perform adaptive evaluation
    print("Performing Adaptive Evaluation using Debevec labels")
    eval_adaptive_ssim = 0.0
    for i in range(all_test_data.shape[0]):
        cur_batch = all_test_data[np.newaxis, i]
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
        _, test_ssim = evaluate_maml(meta_model, loss_func, eval_train, eval_test, cfg.EVAL.BATCH_SIZE, cfg.EVAL.NUM_TASK_TR_ITER, device=device, visualize_flag=True, visualize_dir=evaluation_figure_output_dir)

        eval_adaptive_ssim += test_ssim
    
    eval_adaptive_ssim /= all_test_data.shape[0]

    print("[Evaluation Results] Average Adapted Evaluation SSIM : {:.3f}".format(eval_adaptive_ssim))

    # eval_train, eval_test = dg.sample_batch('meta_test', cfg.EVAL.BATCH_SIZE)


    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Path to directory with outputs from MetaHDR training cycle.')
    parser.add_argument('--use_best', type=bool, default=True, help='Flag as True if evaluation should be done on the best model from training.')
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()

    main(args)