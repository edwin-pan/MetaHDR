import tensorflow as tf

def get_unet(img_rows,img_cols):
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

    conv10 =tf.keras.layers.SeparableConv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.Model(inputs=[inputs], outputs=[conv10])

    # model.compile(optimizer=tfAdam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model