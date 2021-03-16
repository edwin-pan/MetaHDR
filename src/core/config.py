import argparse
from yacs.config import CfgNode as CN

# High-level
cfg = CN()
cfg.OUTPUT_DIR = 'results'
cfg.EXP_NAME = 'default'
cfg.SUMMARY_INTERVAL = 10
cfg.PRINT_INTERVAL = 10
cfg.TEST_PRINT_INTERVAL = 50
cfg.CHECKPOINT_SAVE_PATH = 'checkpoints'

# Loss parameters
cfg.LOSS = CN()
cfg.LOSS.SEP_LOSS = True
cfg.LOSS.LAMBDA = 0.5

# Training parameters
cfg.TRAIN = CN()
cfg.TRAIN.LOSS_FUNC = 'ExpandNetLoss'
cfg.TRAIN.NUM_EXPOSURES = 3
cfg.TRAIN.BATCH_SIZE = 8
cfg.TRAIN.NUM_META_TR_ITER = 200
cfg.TRAIN.NUM_TASK_TR_ITER = 1
cfg.TRAIN.META_LR = 0.001
cfg.TRAIN.TASK_LR = 0.4

# Evaluation parameters
cfg.EVAL = CN()
cfg.EVAL.LOSS_FUNC = 'ExpandNetLoss'
cfg.EVAL.NUM_EXPOSURES = 3
cfg.EVAL.BATCH_SIZE = 8
cfg.EVAL.NUM_TASK_TR_ITER = 1

# Utilities
cfg.UTILS = CN()
cfg.UTILS.GAMMA = 2.2


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for MetaHDR."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()


def update_cfg(cfg_file):
    """
    Update configs with new values from .yaml file
    """
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_file)
    return cfg.clone()


def parse_args():
    """
    Parse input arguments to grab desired configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='cfg file path')

    args = parser.parse_args()
    print(args, end='\n\n')

    cfg_file = args.cfg
    if args.cfg is not None:
        cfg = update_cfg(args.cfg)
    else:
        cfg = get_cfg_defaults()

    return cfg, cfg_file