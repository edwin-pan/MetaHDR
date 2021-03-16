import torch

from src.core.config import parse_args
from src.core.utils import prepare_output_dir

from src.models.metaHDR import train_maml
from src.core.utils import create_logger

def main(cfg, log_dir):
    logger = create_logger(log_dir, phase='train')

    logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    train_maml(cfg, log_dir)

if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg, log_dir = prepare_output_dir(cfg, cfg_file)

    main(cfg, log_dir)