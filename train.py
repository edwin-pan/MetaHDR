import matplotlib.pyplot as plt
from tqdm import tqdm
import GPUtil
import numpy as np
import time
import os

import tensorflow as tf
from skimage.metrics import structural_similarity as ssim

from src.dataset.dataloader import DataGenerator
from src.core.config import parse_args
from src.core.utils import prepare_output_dir
from src.models.metaHDR import MetaHDR
from src.models.metaHDR import outer_train_step, outer_eval_step
from src.models.UNet import get_unet, unet_forward

from src.core.loss import IRLoss

from src.core.loss import temp_mse_loss


def main(cfg):
    # Parameters
    img_H = 512
    img_W = 512

    # Check compute method
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Tensorflow Version: ", tf.__version__)
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        print("Preventing TF pre-allocation of GPU mem")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    # # tf.compat.v1.disable_eager_execution()

    # Define Loss
    # loss_func = IRLoss(img_W, img_H, 0.5).forward
    loss_func = temp_mse_loss

    # Define inner-model
    inner_model = get_unet(img_H, img_W)

    # Define Model 
    model = MetaHDR(loss_func, inner_model, img_width=img_W, img_height=img_H, num_inner_updates=cfg.TRAIN.NUM_TASK_TR_ITER, inner_update_lr=cfg.TRAIN.TASK_LR)
    GPUtil.showUtilization(all=True)

    dl = DataGenerator(num_exposures=cfg.TRAIN.NUM_EXPOSURES)
    train, test = dl.sample_batch('meta_train', cfg.TRAIN.BATCH_SIZE)

    # NOTE: Treating N-way K-shot problem = 2-way 1-shot. Therefore, N*K = N
    _, B, N, H, W, C = train.shape # (2, 8, 2, 512, 512, 3)

    assert cfg.TRAIN.BATCH_SIZE == B, "Data loader should not be changing batchsize!"
    assert img_H == H, "Stated image height and loaded image height are not the same! How bizzare..."
    assert img_W == W, "Stated image width and loaded image width are not the same! How bizzare..."

    # Define Optimizer
    meta_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.TRAIN.META_LR)

    # Set Accuracy tracking
    pre_loss, post_loss, pre_accuracies, post_accuracies = [], [], [], []
    curr_best_performance = 0.0

    time.sleep(1)
    for itr in tqdm(range(cfg.TRAIN.NUM_META_TR_ITER)):
        # Grab batch of data from dataloader
        train, test = dl.sample_batch('meta_train', cfg.TRAIN.BATCH_SIZE)

        # Perform each task-specific training inner-loop
        inp = (train[0], test[0], train[1], test[1])
        result = outer_train_step(inp, model, meta_optimizer, meta_batch_size=cfg.TRAIN.BATCH_SIZE, num_inner_updates=cfg.TRAIN.NUM_TASK_TR_ITER)

        if itr % cfg.SUMMARY_INTERVAL == 0:
            pre_accuracies.append(result[-2])
            post_accuracies.append(result[-1][-1])
            pre_loss.append(result[-4])
            post_loss.append(result[-3][-1])

        if (itr!=0) and itr % cfg.PRINT_INTERVAL == 0:
            # import pdb; pdb.set_trace()
            print_str = 'Iteration %d: pre loss: %.5f, post loss: %.5f, pre-inner-loop train SSIM: %.5f, post-inner-loop test SSIM: %.5f' % (itr, tf.reduce_mean(pre_loss), tf.reduce_mean(post_loss), tf.reduce_mean(pre_accuracies), tf.reduce_mean(post_accuracies))
            print(print_str)
            pre_accuracies, post_accuracies = [], []

        if (itr!=0) and itr % cfg.TEST_PRINT_INTERVAL == 0:
            # sample a batch of validation data and partition into
            # training (input_tr, label_tr) and testing (input_ts, label_ts)
            # ---------------------------
            train, test = dl.sample_batch('meta_val', cfg.TRAIN.BATCH_SIZE)
            inp = (train[0], test[0], train[1], test[1])
            result = outer_eval_step(inp, model, meta_batch_size=cfg.TRAIN.BATCH_SIZE, num_inner_updates=cfg.TRAIN.NUM_TASK_TR_ITER)

            print('[Meta-validation] pre-inner-loop train SSIM: %.5f, meta-validation post-inner-loop test SSIM: %.5f' % (result[-2], result[-1][-1]))

            # Check GPU-Usage
            GPUtil.showUtilization(all=True)

            # Save model if it's our best so far
            if result[-1][-1] > curr_best_performance:
                print(f"Best performance so far! Saving model to {cfg.CHECKPOINT_SAVE_PATH}")
                # model.save(cfg.CHECKPOINT_SAVE_PATH)
                curr_best_performance = result[-1][-1]

    print("Checkpoint")


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)