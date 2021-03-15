import matplotlib.pyplot as plt
from tqdm import tqdm
import GPUtil
import numpy as np
import time
import os

from src.dataset.dataloader import DataGenerator
from src.core.config import parse_args
from src.core.utils import prepare_output_dir

from src.models.torchmetaHDR import train_maml

def main(cfg):
    train_maml(cfg)

if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)