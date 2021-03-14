import tensorflow as tf
import numpy as np
from functools import partial
import GPUtil
import gc

# from tensorflow.image import ssim # Not available in tensorflow/2.1
from skimage.metrics import structural_similarity as ssim

from time import time

from src.models.UNet import get_unet, unet_forward
from src.models.utils import copy_model_fn
from src.core.utils import get_GPU_usage

def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
    """
    MetaHDR's outer training loop handles meta-parameter adjustments, after num_inner_updates number of inner-loop task-specific 
    model updates.
    """

    with tf.GradientTape(persistent=False) as outer_tape:
        result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result
        total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    gradients = outer_tape.gradient(total_losses_ts[-1], model.m.trainable_weights)

    optim.apply_gradients(zip(gradients, model.m.trainable_weights))
    tf.keras.backend.clear_session()
    gc.collect()
    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]
    
    del outputs_tr,outputs_ts # Save space
    return total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts


def outer_eval_step(inp, model, meta_batch_size=25, num_inner_updates=1):
    """
    MetaHDR's outer evaluation step. Performs one forward pass, training the inner loop num_inner_updates number of times.
    """

    result = model(inp, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)

    outputs_tr, outputs_ts, losses_tr_pre, losses_ts, accuracies_tr_pre, accuracies_ts = result

    total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
    total_losses_ts = [tf.reduce_mean(loss_ts) for loss_ts in losses_ts]

    total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
    total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]
    # tf.keras.backend.clear_session()
    return outputs_tr, outputs_ts, total_loss_tr_pre, total_losses_ts, total_accuracy_tr_pre, total_accuracies_ts

class MetaHDR(tf.keras.Model):
    def __init__(self,
                 loss_func,
                 img_width=1,
                 img_height=1,
                 num_inner_updates=1,
                 inner_update_lr=0.4,
                 pretrain_flag=False,
                 model_weights=None):
        """
        Main definition of MetaHDR model. MetaHDR consists of a core UNet architecture wrapped with the MAML framework. The inner loop is
        defined within the MetaHDR class. The outerloop meta-parameter tuning is handled outside (see train.py)
        """
        super(MetaHDR, self).__init__()
        self.width = img_width
        self.height = img_height
        self.inner_update_lr = inner_update_lr
        self.ssim_score = tf.image.ssim
        self.k_shot=1 # This won't change, because we only have 1 example per task
        self.pretrain_flag = pretrain_flag
        self.loss_func = loss_func
        self.non_trainable_layers =  [0,3,6,9,12,16,20,24,28]
        self.up_conv_layers =  [15,19,23,27]
        if self.pretrain_flag:
            print(self.width,self.height)
            self.m = get_unet(self.height, self.width)
            self.m.load_weights(model_weights)
            self.m.build((self.height, self.width, 3))
            print("Loaded weights from: {}".format(model_weights))
        else:
            self.m = get_unet(self.height, self.width)
            self.m.build((self.height, self.width, 3))

    # @tf.function
    def task_inner_loop(self, inp,reuse=True,meta_batch_size=25,num_inner_updates=1):
        # the inner and outer loop data, query set: (input_tr,label_tr)
        input_tr, input_ts, label_tr, label_ts = inp[0], inp[1], inp[2], inp[3]
        task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None
        task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

        with tf.GradientTape(persistent=True) as tape:

            get_GPU_usage("inner pre")
            # tf.keras.backend.clear_session()
            # gc.collect()
            # get_GPU_usage("intermediate post")
            task_output_tr_pre = unet_forward(m, input_tr)
            get_GPU_usage("inner post")

            inner_task_weights = [item for item in self.m.trainable_weights]

            task_loss_tr_pre = self.loss_func(label_tr,task_output_tr_pre)
            grads = tape.gradient(task_loss_tr_pre,inner_task_weights)

            # Apply the gradients
            k=0
            for j in range(len(self.m.layers)):
                # print(j,self.m.layers[j].name)
                if j not in self.non_trainable_layers: # Layers w/ no trainable parameters
                    if j in self.up_conv_layers: # Up-conv layers
                        self.m.layers[j].kernel=self.m.layers[j].kernel - self.inner_update_lr*grads[k]
                        self.m.layers[j].bias=self.m.layers[j].bias - self.inner_update_lr*grads[k+1]
                        k+=2
                        self.m.layers[j].trainable=True
                    else: # Regular conv layers
                        self.m.layers[j].depthwise_kernel=self.m.layers[j].depthwise_kernel - self.inner_update_lr*grads[k]
                        self.m.layers[j].pointwise_kernel=self.m.layers[j].pointwise_kernel - self.inner_update_lr*grads[k+1]
                        self.m.layers[j].bias=self.m.layers[j].bias - self.inner_update_lr*grads[k+2]
                        k+=3
                        self.m.layers[j].trainable=True

            output_ts = unet_forward(m, input_ts)
            loss_ts = self.loss_func(label_ts,output_ts)
            task_outputs_ts.append(output_ts)
            task_losses_ts.append(loss_ts)
            
            # Now revert the gradients
            k=0
            for j in range(len(self.m.layers)):
                # print(j,self.m.layers[j].name)
                if j not in self.non_trainable_layers: # Layers w/ no trainable parameters
                    if j in self.up_conv_layers: # Up-conv layers
                        self.m.layers[j].kernel=inner_task_weights[k]
                        self.m.layers[j].bias=inner_task_weights[k+1]
                        k+=2
                        self.m.layers[j].trainable=True
                    else: # Regular conv layers
                        self.m.layers[j].depthwise_kernel=inner_task_weights[k]
                        self.m.layers[j].pointwise_kernel=inner_task_weights[k+1]
                        self.m.layers[j].bias=inner_task_weights[k+2]
                        k+=3
                        self.m.layers[j].trainable=True
            
        # Compute accuracies from output predictions
        task_accuracy_tr_pre = tf.reduce_mean(self.ssim_score(label_tr,task_output_tr_pre, 1.0))
        
        for j in range(num_inner_updates):
            task_accuracies_ts.append(self.ssim_score(label_ts,task_outputs_ts[j], 1.0))

        task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

        return task_output

    # @tf.function
    def call(self,
             inp,
             meta_batch_size=25,
             num_inner_updates=1):
        input_tr, input_ts, label_tr, label_ts = inp
        out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
        task_inner_loop_partial = partial(self.task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial, elems=(input_tr, input_ts, label_tr, label_ts),
                        dtype=out_dtype,
                        parallel_iterations=meta_batch_size)
        return result