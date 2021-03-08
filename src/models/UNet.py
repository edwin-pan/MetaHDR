import tensorflow as tf

def get_unet(img_rows,img_cols):
    """
        UNet architecture used.

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
    """

    inputs = tf.keras.layers.Input((img_rows, img_cols, 3))
    # tf.keras.layers.SeparableConv2D(fil)
    conv1 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.SeparableConv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.SeparableConv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 =tf.keras.layers.SeparableConv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 =tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 =tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 =tf.keras.layers.SeparableConv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 =tf.keras.layers.SeparableConv2D(1, (1, 1), activation='relu')(conv9) # Don't use sigmoid since we are performing "regression"

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])

    # model.compile(optimizer=tfAdam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model