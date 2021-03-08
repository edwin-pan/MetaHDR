import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os

import tensorflow as tf
import segmentation_models as sm

from src.dataset.dataloader import DataGenerator
from src.core.config import parse_args
from src.core.utils import prepare_output_dir
from src.models.metaHDR import MetaHDR
from src.core.loss import IRLoss


def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
    """
    MetaHDR's outer training loop handles meta-parameter adjustments, after num_inner_updates number of inner-loop task-specific 
    model updates.
    """
    # note here, outer tape constructed to watch all model.trainable_variables!
    # inner_loop is called in model(...)
    # no need to do persistent, since only 1 outer_tape.gradient needs to be called
    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result
        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    # dont need to update self.inner_update_lr_dict,
    # since learn rate is part of the model.training_variables
    gradients = outer_tape.gradient(total_losses_ts[-1], model.m.trainable_weights)

    # this will update ALL PARAMETERS, including the LEARN RATE!
    # rather than manual gradient descent, Adam (adaptive grad descent) used to update params
    optim.apply_gradients(zip(gradients, model.m.trainable_weights))

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]
    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):

    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]
    # tf.keras.backend.clear_session()
    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def temp_mse_loss(y_true, y_pred):
    # return tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def main(cfg):
    # Parameters
    img_H = 512
    img_W = 512

    # Check compute method
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Define Loss
    # loss_func = IRLoss(img_W, img_H, 0.5).forward
    loss_func = temp_mse_loss

    # Define Model 
    model = MetaHDR(loss_func, img_width=img_W, img_height=img_H, num_inner_updates=cfg.TRAIN.NUM_TASK_TR_ITER, inner_update_lr=cfg.TRAIN.TASK_LR)
    
    dl = DataGenerator()
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
            print_str = 'Iteration %d: pre loss: %.5f, post loss: %.5f, pre-inner-loop train SSIM: %.5f, post-inner-loop test SSIM: %.5f' % (itr, np.mean(pre_loss), np.mean(post_loss), np.mean(pre_accuracies), np.mean(post_accuracies))
            print(print_str)
            pre_accuracies, post_accuracies = [], []

        if (itr!=0) and itr % cfg.TEST_PRINT_INTERVAL == 0:
            # sample a batch of validation data and partition into
            # training (input_tr, label_tr) and testing (input_ts, label_ts)
            # ---------------------------
            train, test = dl.sample_batch('meta_val', cfg.TRAIN.BATCH_SIZE)
            inp = (train[0], test[0], train[1], test[1])
            result = outer_train_step(inp, model, meta_optimizer, meta_batch_size=cfg.TRAIN.BATCH_SIZE, num_inner_updates=cfg.TRAIN.NUM_TASK_TR_ITER)

    print("Checkpoint")


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)