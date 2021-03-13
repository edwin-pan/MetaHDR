import tensorflow as tf

def get_unet(img_cols,img_rows):
    """
        UNet architecture used.

        ==== LAYERS ====
        0 input_1
        1 separable_conv2d
        2 separable_conv2d_1
        3 max_pooling2d
        4 separable_conv2d_2
        5 separable_conv2d_3
        6 max_pooling2d_1
        7 separable_conv2d_4
        8 separable_conv2d_5
        9 max_pooling2d_2
        10 separable_conv2d_6
        11 separable_conv2d_7
        12 max_pooling2d_3
        13 separable_conv2d_8
        14 separable_conv2d_9
        15 conv2d_transpose
        16 concatenate
        17 separable_conv2d_10
        18 separable_conv2d_11
        19 conv2d_transpose_1
        20 concatenate_1
        21 separable_conv2d_12
        22 separable_conv2d_13
        23 conv2d_transpose_2
        24 concatenate_2
        25 separable_conv2d_14
        26 separable_conv2d_15
        27 conv2d_transpose_3
        28 concatenate_3
        29 separable_conv2d_16
        30 separable_conv2d_17
        31 separable_conv2d_18
        ================ 
        
        ==== TRAINABLE ====
        0 separable_conv2d_19/depthwise_kernel:0
        1 separable_conv2d_19/pointwise_kernel:0
        2 separable_conv2d_19/bias:0
        3 separable_conv2d_20/depthwise_kernel:0
        4 separable_conv2d_20/pointwise_kernel:0
        5 separable_conv2d_20/bias:0
        6 separable_conv2d_21/depthwise_kernel:0
        7 separable_conv2d_21/pointwise_kernel:0
        8 separable_conv2d_21/bias:0
        9 separable_conv2d_22/depthwise_kernel:0
        10 separable_conv2d_22/pointwise_kernel:0
        11 separable_conv2d_22/bias:0
        12 separable_conv2d_23/depthwise_kernel:0
        13 separable_conv2d_23/pointwise_kernel:0
        14 separable_conv2d_23/bias:0
        15 separable_conv2d_24/depthwise_kernel:0
        16 separable_conv2d_24/pointwise_kernel:0
        17 separable_conv2d_24/bias:0
        18 separable_conv2d_25/depthwise_kernel:0
        19 separable_conv2d_25/pointwise_kernel:0
        20 separable_conv2d_25/bias:0
        21 separable_conv2d_26/depthwise_kernel:0
        22 separable_conv2d_26/pointwise_kernel:0
        23 separable_conv2d_26/bias:0
        24 separable_conv2d_27/depthwise_kernel:0
        25 separable_conv2d_27/pointwise_kernel:0
        26 separable_conv2d_27/bias:0
        27 separable_conv2d_28/depthwise_kernel:0
        28 separable_conv2d_28/pointwise_kernel:0
        29 separable_conv2d_28/bias:0
        30 conv2d_transpose_4/kernel:0
        31 conv2d_transpose_4/bias:0
        32 separable_conv2d_29/depthwise_kernel:0
        33 separable_conv2d_29/pointwise_kernel:0
        34 separable_conv2d_29/bias:0
        35 separable_conv2d_30/depthwise_kernel:0
        36 separable_conv2d_30/pointwise_kernel:0
        37 separable_conv2d_30/bias:0
        38 conv2d_transpose_5/kernel:0
        39 conv2d_transpose_5/bias:0
        40 separable_conv2d_31/depthwise_kernel:0
        41 separable_conv2d_31/pointwise_kernel:0
        42 separable_conv2d_31/bias:0
        43 separable_conv2d_32/depthwise_kernel:0
        44 separable_conv2d_32/pointwise_kernel:0
        45 separable_conv2d_32/bias:0
        46 conv2d_transpose_6/kernel:0
        47 conv2d_transpose_6/bias:0
        48 separable_conv2d_33/depthwise_kernel:0
        49 separable_conv2d_33/pointwise_kernel:0
        50 separable_conv2d_33/bias:0
        51 separable_conv2d_34/depthwise_kernel:0
        52 separable_conv2d_34/pointwise_kernel:0
        53 separable_conv2d_34/bias:0
        54 conv2d_transpose_7/kernel:0
        55 conv2d_transpose_7/bias:0
        56 separable_conv2d_35/depthwise_kernel:0
        57 separable_conv2d_35/pointwise_kernel:0
        58 separable_conv2d_35/bias:0
        59 separable_conv2d_36/depthwise_kernel:0
        60 separable_conv2d_36/pointwise_kernel:0
        61 separable_conv2d_36/bias:0
        62 separable_conv2d_37/depthwise_kernel:0
        63 separable_conv2d_37/pointwise_kernel:0
        64 separable_conv2d_37/bias:0
        ============

    """

    inputs = tf.keras.layers.Input((img_rows, img_cols, 3))
    # tf.keras.layers.SeparableConv2D(fil)
    conv1 = tf.keras.layers.SeparableConv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.SeparableConv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same')(up8)
    conv8 =tf.keras.layers.SeparableConv2D(16, (3, 3), activation='relu', padding='same')(conv8)

    up9 =tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 =tf.keras.layers.SeparableConv2D(8, (3, 3), activation='relu', padding='same')(up9)
    conv9 =tf.keras.layers.SeparableConv2D(8, (3, 3), activation='relu', padding='same')(conv9)

    conv10 =tf.keras.layers.SeparableConv2D(3, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])
    
    return model