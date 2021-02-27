import pandas as pd
import matplotlib.pyplot as plt
from dataloader import *
import os
from tqdm import tqdm

# images,
# k_shot=2
images,labels = d.sample_batch('meta_train',8)
b,n,k,w,h,c = images.shape
print(b,n,k,w,h,c)
images = images.reshape((b,n*k,w,h,c))
print("images: {}".format(images.shape))
labels = labels.reshape((b,n*k,w,h,1))

input_tr  = images[:,:k_shot]
label_tr = labels[:,:k_shot]
input_ts = images[:,k_shot:]
label_ts = labels[:,k_shot:]

input_tr = preprocess_input(input_tr)
input_tr = tf.convert_to_tensor(input_tr,tf.float32)
input_ts = preprocess_input(input_ts)
input_ts = tf.convert_to_tensor(input_ts,tf.float32)

label_tr = tf.convert_to_tensor(label_tr,tf.float32)
label_ts = tf.convert_to_tensor(label_ts,tf.float32)
inp = (input_tr, input_ts, label_tr, label_ts)
# result = model(inp, meta_batch_size=16, num_inner_updates=1)