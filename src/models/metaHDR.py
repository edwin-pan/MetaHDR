import tensorflow as tf
import numpy as np
from functools import partial
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score
# from tensorflow.image import ssim # Not available in tensorflow/2.1
from skimage.metrics import structural_similarity as ssim

from time import time

from src.models.UNet import get_unet
from src.models.utils import copy_model_fn

class MetaHDR(tf.keras.Model):
    def __init__(self,
                 loss_func,
                 img_width=1,
                 img_height=1,
                 num_inner_updates=1,
                 inner_update_lr=0.4,
                 backbone='mobilenet',
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
        self.get_preprocessing = get_preprocessing
        self.ssim_score = partial(ssim, multichannel=True)
        self.BACKBONE = backbone
        self.k_shot=1 # This won't change, because we only have 1 example per task
        self.pretrain_flag = pretrain_flag
        self.loss_func = loss_func
        self.preprocess_input = get_preprocessing(self.BACKBONE)
        if self.pretrain_flag:
            print(self.width,self.height)
            self.m = get_unet(self.width,self.height)
            self.m.load_weights(model_weights)
            print("Loaded weights from: {}".format(model_weights))
        else:
            self.m = get_unet(self.width,self.height)


        # outputs_ts[i] and losses_ts_post[i] are the output and loss after i+1 inner gradient updates
        losses_tr_pre, outputs_tr, losses_ts_post, outputs_ts = [], [], [], []
        accuracies_tr_pre, accuracies_ts = [], []

        # for each loop in the inner training loop
        outputs_ts = [[]]*num_inner_updates
        losses_ts_post = [[]]*num_inner_updates
        accuracies_ts = [[]]*num_inner_updates

#     @tf.function
    def call(self,
             inp,
             meta_batch_size=25,
             num_inner_updates=1):
        def task_inner_loop(inp,reuse=True,meta_batch_size=25,num_inner_updates=1):
            '''
            '''
            # the inner and outer loop data
            # query set: (input_tr,label_tr)
            input_tr, input_ts, label_tr, label_ts = inp

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []

            ######################################
            ############### MAML #################
            with tf.GradientTape(persistent=True) as tape:

                copied_model = copy_model_fn(self.m,self.width,self.height)
                task_output_tr_pre = copied_model(input_tr)# logits

                # copied_model.summary()

                task_loss_tr_pre = self.loss_func(label_tr,task_output_tr_pre)
                grads = tape.gradient(task_loss_tr_pre,copied_model.trainable_weights)
                
                k=0
                for j in range(len(self.m.layers)):
                    # print(j,self.m.layers[j].name)
                    if j not in [0,3,6,9,12,16,20,24,28]: # Layers w/ no trainable parameters
                        if j in [15,19,23,27]: # Up-conv layers
                            copied_model.layers[j].kernel=self.m.layers[j].kernel - self.inner_update_lr*grads[k]
                            copied_model.layers[j].bias=self.m.layers[j].bias - self.inner_update_lr*grads[k+1]
                            k+=2
                            copied_model.layers[j].trainable=True
                        else: # Regular conv layers
                            copied_model.layers[j].depthwise_kernel=self.m.layers[j].depthwise_kernel - self.inner_update_lr*grads[k]
                            copied_model.layers[j].pointwise_kernel=self.m.layers[j].pointwise_kernel - self.inner_update_lr*grads[k+1]
                            copied_model.layers[j].bias=self.m.layers[j].bias - self.inner_update_lr*grads[k+2]
                            k+=3
                            copied_model.layers[j].trainable=True

                output_ts = copied_model(input_ts)
                loss_ts = self.loss_func(label_ts,output_ts)
                # tape.stop_recording()
                del copied_model
                task_outputs_ts.append(output_ts)
                # print("task_outputs_ts {}".format(task_outputs_ts))#(B,5)
                task_losses_ts.append(loss_ts)
            # Compute accuracies from output predictions

            # # DEBUG
            # import pdb; pdb.set_trace()

            task_accuracy_tr_pre = 0
            N = label_tr.shape[0]
            for i in range(N):
                task_accuracy_tr_pre += self.ssim_score(label_tr[0].numpy(),task_output_tr_pre[i].numpy())
            task_accuracy_tr_pre /= N

            for j in range(num_inner_updates):
                task_accuracies_ts.append(self.ssim_score(label_ts.numpy().squeeze(),task_outputs_ts[j].numpy().squeeze()))

            # task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

            del task_output_tr_pre,task_outputs_ts # Save space
            task_output = [task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

            return task_output

        input_tr, input_ts, label_tr, label_ts = inp
        # out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
        out_dtype = [tf.float32, [tf.float32]*num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
        task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial, elems=(input_tr, input_ts, label_tr, label_ts),
                        dtype=out_dtype,
                        parallel_iterations=meta_batch_size)
        return result