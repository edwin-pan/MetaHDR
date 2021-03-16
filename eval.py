import torch
import argparse
from os import path as osp
import learn2learn as l2l

from src.core.config import update_cfg, get_cfg_defaults
from src.models.UNet import UNet
from src.models.metaHDR import evaluate_maml
from src.dataset.dataloader import DataGenerator
from src.core.loss import get_loss_func

def main(args):
    print("--- Evaluating on held-out portion of dataset ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    source_directory = args.model_dir
    use_best_flag = args.use_best

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
    meta_model = l2l.algorithms.MAML(model)
    print(f"Loading pre-trained model from --> {model_path}")
    meta_model.load_state_dict(checkpoint['unet_state_dict'])
    print(f"Successfully loaded pre-trained model from --> {model_path}")

    # Grad test data
    dg = DataGenerator(num_exposures=cfg.EVAL.NUM_EXPOSURES)
    eval_train, eval_test = dg.sample_batch('meta_test', cfg.EVAL.BATCH_SIZE)

    evaluate_maml(meta_model, loss_func, eval_train, eval_test, cfg.EVAL.BATCH_SIZE, cfg.EVAL.NUM_TASK_TR_ITER, device=device)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='Path to directory with outputs from MetaHDR training cycle.')
    parser.add_argument('--use_best', type=bool, default=True, help='Flag as True if evaluation should be done on the best model from training.')
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()

    main(args)