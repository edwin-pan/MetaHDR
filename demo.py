import argparse
import learn2learn as l2l
import torch
import skimage.io as io
import numpy as np
import glob
import os

from src.models.UNet import UNet
from src.models.metaHDR import evaluate_maml, evaluate_single_maml
from src.core.config import update_cfg, get_cfg_defaults
from src.models.UNet import UNet
from src.core.loss import get_loss_func

def main(args):
    print("--- Running MetaHDR Demo ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ----- SETUP -----
    # Parse args
    images_dir = args.input_folder
    output_dir = args.output_folder
    
    # Create output folder if doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Parse provided images
    LDR_input_dir = os.path.join(images_dir, 'LDR')
    HDR_input_dir = os.path.join(images_dir, 'HDR')
    
    # Check that folders exist
    assert os.path.isdir(LDR_input_dir), f"{images_dir}/LDR doesn't exist!"
    assert os.path.isdir(HDR_input_dir), f"{images_dir}/HDR doesn't exist!"
    
    # Grab all LDR, HDR
    supported_filetypes = ['.jpg', '.png']
    input_LDR_fnames, input_HDR_fnames = [], []
    
    for ending in supported_filetypes:
        input_LDR_fnames.extend(glob.glob(os.path.join(LDR_input_dir, '*'+ending)))
        input_HDR_fnames.extend(glob.glob(os.path.join(HDR_input_dir, '*'+ending)))
    
    # Check that there are the same number of files in LDR and HDR
    assert len(input_HDR_fnames) == len(input_LDR_fnames), "There should be an HDR image for each provided LDR exposure"
    num_exposures = len(input_LDR_fnames)
    
    # Load data
    LDR_inputs , HDR_inputs = [], []
    for LDR_fname, HDR_fname in zip(input_LDR_fnames, input_HDR_fnames):
        LDR_inputs.append(io.imread(LDR_fname).astype(np.float64)/255)
        HDR_inputs.append(io.imread(HDR_fname).astype(np.float64)/255)
    LDR_inputs = np.array(LDR_inputs)
    HDR_inputs = np.array(HDR_inputs)
    
    # Instantiate MetaHDR model
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()
    
    # Make sure loss_func from config is valid, then get it
    assert cfg.EVAL.LOSS_FUNC in ['ExpandNetLoss', 'HaarLoss', 'LPIPSLoss', 'LPIPSLoss_L2', 'SSIMLoss'], f"[CONFIG] evaluation loss function '{cfg.EVAL.LOSS_FUNC}' not valid"
    loss_func = get_loss_func(cfg.EVAL.LOSS_FUNC)
    
    # Load weights
    model_path = './data/demo_weights/model_demo.pth.tar'
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
    
    # ----- FORWARD-PASS -----
    if num_exposures == 1:
        # If only provided 1 LDR image, perform singe-shot
        print("[MetaHDR] Single exposure provided. Running without adaptation.")
        _, test_ssim, test_psnr = evaluate_single_maml(meta_model, loss_func, LDR_inputs, HDR_inputs, 0, device=device, visualize_flag=True, visualize_dir=output_dir)
        print(f"[Single-Shot {0:03d}] SSIM: {test_ssim}, PSNR: {test_psnr}")
    else:
        # If provided more than 1 LDR image, adapt
        print("[MetaHDR] Multiple exposures provided. Running with adaptation.")
        train_inp = LDR_inputs[:2]
        train_lab = HDR_inputs[:2]
        test_inp = LDR_inputs[np.newaxis, -1]
        test_lab = HDR_inputs[np.newaxis, -1]
        training = np.stack((train_inp, train_lab))
        testing = np.stack((test_inp, test_lab))

        _, test_ssim, test_psnr = evaluate_maml(meta_model, loss_func, training, testing, 0, cfg.EVAL.NUM_TASK_TR_ITER, device=device, model_type=cfg.TRAIN.MODEL, visualize_flag=True, visualize_dir=output_dir)
        print(f"[Single-Shot {0:03d}] SSIM: {test_ssim}, PSNR: {test_psnr}")
        
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='Path to input images')
    parser.add_argument('--output_folder', type=str, help='Path to output images')
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()

    main(args)