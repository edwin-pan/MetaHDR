import tensorflow as tf
from functools import partial
from segmentation_models import get_preprocessing
from segmentation_models.metrics import iou_score

from time import time

from lib.models.UNet import get_unet
from lib.models.utils import copy_model_fn

class MAML_UNet(tf.keras.Model):
    def __init__(self,
                 model_weights,
                 width=1,
                 height=1,
                 num_inner_updates=1,
                 inner_update_lr=0.4,
                 k_shot=5,
                 backbone='mobilenet',
                 pretrain=False):
        '''
        '''
        super(MAML_UNet, self).__init__()
        self.width = width
        self.height = height
        self.inner_update_lr = inner_update_lr
        self.get_preprocessing = get_preprocessing
        self.iou_score = iou_score
        self.BACKBONE = backbone
        self.pretrain = pretrain
        self.loss_func = total_loss
        self.preprocess_input = get_preprocessing(self.BACKBONE)
        if self.pretrain:
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

        self.kernels = []
        self.biases = []
        '''
        for j in range(len(self.m.layers)):
            # print(j,m.layers[j].name,copied_model.layers[j].name)
            
            if j not in [0,3,6,9,12,16,20,24,28]:
                # print(j,copied_model.layers[j].name)
                # print(j,k,copied_model.layers[j].kernel.shape,m.layers[j].kernel.shape,grads[k].shape)
                # print(j,k+1,copied_model.layers[j].bias.shape,m.layers[j].bias.shape,grads[k+1].shape)
                # print(j,copied_model.layers[j].bias.name,copied_model.layers[j].bias.shape)
                # print(j,copied_model.layers[j].kernel.name,m.layers[j].kernel.shape)
                # print(grads[k].shape)
                self.kernels.append(self.m.layers[j].kernel)
                self.biases.append(self.m.layers[j].bias)
        '''

        # self.copied_model = copy_model_fn(self.m,self.width,self.height)
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

            # weights corresponds to the initial weights in MAML (i.e. the meta-parameters)
            # weights = self.w

            # the predicted outputs, loss values, and accuracy for the pre-update model (with the initial weights)
            # evaluated on the inner loop training data
            task_output_tr_pre, task_loss_tr_pre, task_accuracy_tr_pre = None, None, None

            # lists to keep track of outputs, losses, and accuracies of test data for each inner_update
            # where task_outputs_ts[i], task_losses_ts[i], task_accuracies_ts[i] are the output, loss, and accuracy
            # after i+1 inner gradient updates
            task_outputs_ts, task_losses_ts, task_accuracies_ts = [], [], []
            # model = Unet(self.BACKBONE, encoder_weights=None)
            # for i in range(len(model.trainable_weights)):
            #     model.trainable_weights[i].assign(weights[i])
                
            # model = copy_model_fn(self.m,input_tr) 
            # model(input_tr)
            # copy_model(input_tr)
            ######################################
            ############### MAML #################
            # copied_model = copy_model_fn(self.m,self.width,self.height)
            
            with tf.GradientTape(persistent=True) as tape:
                # print(input_tr.shape)
                # tape.watch(model.trainable_weights)
                copied_model = copy_model_fn(self.m,self.width,self.height)
#                 tape.watch(copied_model.trainable_weights)
                task_output_tr_pre = copied_model(input_tr)# logits
                # print(len(self.copied_model.layers))
                task_loss_tr_pre = self.loss_func(label_tr,task_output_tr_pre)
                grads = tape.gradient(task_loss_tr_pre,copied_model.trainable_weights)
                
                '''
                for j in range(len(self.m.layers)):
                    # print(j,m.layers[j].name,copied_model.layers[j].name)
                    
                    if j not in [0,3,6,9,12,16,20,24,28]:
                        # print(j,len(self.m.layers),len(self.copied_model.layers),len(grads),self.m.layers[j].name,self.copied_model.layers[j].name)
                        # print(j,k,copied_model.layers[j].kernel.shape,m.layers[j].kernel.shape,grads[k].shape)
                        # print(j,k+1,copied_model.layers[j].bias.shape,m.layers[j].bias.shape,grads[k+1].shape)
                        # print(j,copied_model.layers[j].bias.name,copied_model.layers[j].bias.shape)
                        # print(j,copied_model.layers[j].kernel.name,m.layers[j].kernel.shape)
                        # print(grads[k].shape)
                        copied_model.layers[j].kernel = self.m.layers[j].kernel - self.inner_update_lr*grads[k]
                        copied_model.layers[j].bias = self.m.layers[j].bias - self.inner_update_lr*grads[k+1]
                        k+=2
                '''
#                 for i in range(len(copied_model.trainable_weights)):
#                     copied_model.trainable_weights[i] = self.m.trainable_weights[i]-self.inner_update_lr*grads[i]
                k=0
                for j in range(len(self.m.layers)):
                    # print(j,m.layers[j].name)
                    
                    if j not in [0,3,6,9,12,16,20,24,28]:
                        if j in [15,19,23,27]:
                            # print(j,m.layers[j].kernel.name,m.layers[j].bias.name)
#                             print(copied_model.layers[j].kernel.shape)
                            copied_model.layers[j].kernel=self.m.layers[j].kernel - self.inner_update_lr*grads[k]
                            copied_model.layers[j].bias=self.m.layers[j].bias - self.inner_update_lr*grads[k+1]
                            k+=2
#                             print(copied_model.layers[j].kernel.shape)
                            copied_model.layers[j].trainable=True
                        else:
                            # print(j,m.layers[j].name,m.layers[j].depthwise_kernel.name,m.layers[j].pointwise_kernel.name,m.layers[j].bias.name)
                            copied_model.layers[j].depthwise_kernel=self.m.layers[j].depthwise_kernel - self.inner_update_lr*grads[k]
                            copied_model.layers[j].pointwise_kernel=self.m.layers[j].pointwise_kernel - self.inner_update_lr*grads[k+1]
                            copied_model.layers[j].bias=self.m.layers[j].bias - self.inner_update_lr*grads[k+2]
                            k+=3
                            copied_model.layers[j].trainable=True

                output_ts = copied_model(input_ts)
                loss_ts = self.loss_func(label_ts,output_ts)
                # tape.stop_recording()
#                 del copied_model
                task_outputs_ts.append(output_ts)
                # print("task_outputs_ts {}".format(task_outputs_ts))#(B,5)
                task_losses_ts.append(loss_ts)
            # Compute accuracies from output predictions
            task_accuracy_tr_pre = self.iou_score(label_tr,task_output_tr_pre)

            for j in range(num_inner_updates):
                task_accuracies_ts.append(self.iou_score(label_ts,task_outputs_ts[j]))

            task_output = [task_output_tr_pre, task_outputs_ts, task_loss_tr_pre, task_losses_ts, task_accuracy_tr_pre, task_accuracies_ts]

            return task_output

        out_dtype = [tf.float32, [tf.float32]*num_inner_updates, tf.float32, [tf.float32]*num_inner_updates]
        out_dtype.extend([tf.float32, [tf.float32]*num_inner_updates])
        task_inner_loop_partial = partial(task_inner_loop, meta_batch_size=meta_batch_size, num_inner_updates=num_inner_updates)
        result = tf.map_fn(task_inner_loop_partial,
                        elems=(input_tr, input_ts, label_tr, label_ts),
                        dtype=out_dtype,
                        parallel_iterations=meta_batch_size)
        return result



def outer_train_step(inp, model, optim, meta_batch_size=25, num_inner_updates=1):
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
  # tf.keras.backend.clear_session()

  total_loss_tr_pre = tf.reduce_mean(losses_tr_pre)
  total_accuracy_tr_pre = tf.reduce_mean(accuracies_tr_pre)
  total_accuracies_ts = [tf.reduce_mean(accuracy_ts) for accuracy_ts in accuracies_ts]
  # tf.keras.backend.clear_session()
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

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# t0 = time()
# result = outer_train_step(inp, model, optimizer, meta_batch_size=8, num_inner_updates=1)
# print("Time: {}".format(time()-t0))