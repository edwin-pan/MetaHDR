# import tensorflow as tf
# import numpy as np
# import scipy.stats as st

# # @tf.function
# def temp_mse_loss(y_true, y_pred):
#     """ Debugging MSE loss """
#     return tf.keras.losses.mean_squared_error(y_true, y_pred)

# def temp_ssim_loss(y_true, y_pred):
#     """ Debugging SSIM loss """
#     N = y_true.shape[0]
#     loss = 0
#     for i in range(N):
#         loss += ssim(y_true[i].numpy(), y_pred[i].numpy(), multichannel=True)
#     loss /= N
#     return tf.convert_to_tensor(1-loss)


# # TODO: Figure our how this loss works (https://arxiv.org/abs/1710.07480)
# class IRLoss():
#     def __init__(self, img_width, img_height, lambda_ir, threshold=0.05):
#     # For masked loss, only using information near saturated image regions
#         self.thr = threshold # Threshold for blending
#         self.lambda_ir = lambda_ir
#         self.sx = img_width
#         self.sy = img_height
#         self.separated_flag = False # TODO: Why did the authors put two versions?


#     def create_mask(self, y_):
#         self.msk = tf.reduce_max(y_, reduction_indices=[3])
#         self.msk = tf.minimum(1.0, tf.maximum(0.0, self.msk-1.0+self.thr)/self.thr)
#         self.msk = tf.reshape(self.msk, [-1, self.sy, self.sx, 1])
#         self.msk = tf.tile(self.msk, [1,1,1,3])


#     def forward(self, x, y, y_, eps):
#         if self.separated_flag:
#             return self.forward_separated(x,y,y_,eps)
#         else:
#             return self.forward_combined(x,y,y_,eps)


#     def forward_separated(self, x, y, y_, eps):
#         """
#         Loss separated into illumination and reflectance terms.

#         Args:
#             x: 
#             y: 
#             y_:
#             eps:
#         """
#         y_log_ = tf.log(y_+eps)
#         x_log = tf.log(tf.pow(x, 2.0)+eps)

#         # Luminance
#         lum_kernel = np.zeros((1, 1, 3, 1))
#         lum_kernel[:, :, 0, 0] = 0.213
#         lum_kernel[:, :, 1, 0] = 0.715
#         lum_kernel[:, :, 2, 0] = 0.072
#         y_lum_lin_ = tf.nn.conv2d(y_, lum_kernel, [1, 1, 1, 1], padding='SAME')
#         y_lum_lin = tf.nn.conv2d(tf.exp(y)-eps, lum_kernel, [1, 1, 1, 1], padding='SAME')
#         x_lum_lin = tf.nn.conv2d(x, lum_kernel, [1, 1, 1, 1], padding='SAME')

#         # Log luminance
#         y_lum_ = tf.log(y_lum_lin_ + eps)
#         y_lum = tf.log(y_lum_lin + eps)
#         x_lum = tf.log(x_lum_lin + eps)

#         # Gaussian kernel
#         nsig = 2
#         filter_size = 13
#         interval = (2*nsig+1.)/(filter_size)
#         ll = np.linspace(-nsig-interval/2., nsig+interval/2., filter_size+1)
#         kern1d = np.diff(st.norm.cdf(ll))
#         kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
#         kernel = kernel_raw/kernel_raw.sum()

#         # Illumination, approximated by means of Gaussian filtering
#         weights_g = np.zeros((filter_size, filter_size, 1, 1))
#         weights_g[:, :, 0, 0] = kernel
#         y_ill_ = tf.nn.conv2d(y_lum_, weights_g, [1, 1, 1, 1], padding='SAME')
#         y_ill = tf.nn.conv2d(y_lum, weights_g, [1, 1, 1, 1], padding='SAME')
#         x_ill = tf.nn.conv2d(x_lum, weights_g, [1, 1, 1, 1], padding='SAME')

#         # Reflectance
#         y_refl_ = y_log_ - tf.tile(y_ill_, [1,1,1,3])
#         y_refl = y - tf.tile(y_ill, [1,1,1,3])
#         x_refl = x - tf.tile(x_ill, [1,1,1,3])

#         cost =              tf.reduce_mean( ( self.lambda_ir*tf.square( tf.subtract(y_ill, y_ill_) ) + (1.0-self.lambda_ir)*tf.square( tf.subtract(y_refl, y_refl_) ) )*self.msk )
#         cost_input_output = tf.reduce_mean( ( self.lambda_ir*tf.square( tf.subtract(x_ill, y_ill_) ) + (1.0-self.lambda_ir)*tf.square( tf.subtract(x_refl, y_refl_) ) )*self.msk )


#     def forward_combined(self, x, y, y_, eps):
#         """
#         Loss with both illumination and reflectance terms combined.

#         Args:
#             x: 
#             y: 
#             y_:
#             eps:
#         """
#         cost =              tf.reduce_mean( tf.square( tf.subtract(y, tf.log(y_+eps) )*self.msk ) )
#         cost_input_output = tf.reduce_mean( tf.square( tf.subtract(tf.log(y_+eps), tf.log(tf.pow(x, 2.0)+eps) )*self.msk ) )