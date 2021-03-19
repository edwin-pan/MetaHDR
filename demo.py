import argparse
import learn2learn as l2l
import torch
import os

from src.models.UNet import UNet
from src.models.metaHDR import evaluate_maml, evaluate_single_maml
from src.core.config import update_cfg, get_cfg_defaults
from src.models.UNet import UNet
from src.core.loss import get_loss_func

def main(args):
    print("--- Running MetaHDR Demo ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse args
    images_dir = args.input_folder
    output_dir = args.output_folder

    # Create output folder if doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Load provided images

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
    checkpoint = torch.load([model_path])
    best_performance = checkpoint['performance']
    best_epoch = checkpoint['epoch']
    print(f"During training: Best Epoch: {best_epoch}, Best SSIM: {best_performance}")

    # Define blank model to load weights into
    model = UNet(in_size=3, out_size=3, num_filters=8).double().to(device)
    meta_model = l2l.algorithms.MAML(model, lr=cfg.EVAL.TASK_LR)
    print(f"Loading pre-trained model from --> {model_path}")
    meta_model.load_state_dict(checkpoint['unet_state_dict'])
    print(f"Successfully loaded pre-trained model from --> {model_path}")

    # If only provided 1 LDR image, perform singe-shot

    # If provided more than 1 LDR image, adapt
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, help='Path to input images')
    parser.add_argument('--output_folder', type=str, help='Path to output images')
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()

    main(args)