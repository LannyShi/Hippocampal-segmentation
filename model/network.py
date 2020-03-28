from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.layers.core import Layer
from keras.layers import multiply, concatenate, Reshape, Permute, Dense, BatchNormalization, Activation, add, Dropout
import tensorflow as tf
from attention_module import *
from CAHFEM import *
weight_decay = 1e-4

smooth = 1.

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_unet():

    img_input = Input(shape=(32, 32, 3), name='image') #（32,32,1）
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True, data_format='channels_last')(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(x1)
    ds1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)

    # Block 2
    # input is of size : 16 x 16 x 64
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x2)
    ds2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)

    # Block 3
    # input is of size : 8 x 8 x 128
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(ds2)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x3)
    ds3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    # Block 4
    # input is of size ：4×4×256
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds3)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x4)
    ds4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)

    # Block 5
    # input is of size ：2×2×512
    x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(ds4)
    x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x5)

    # Upsampling 1
    us1 = concatenate([UpSampling2D(size=(2, 2))(x5), x4])
    x6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv1', trainable=True)(us1)
    x6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv2', trainable=True)(x6)

    # Upsampling 2
    us2 = concatenate([UpSampling2D(size=(2, 2))(x6), x3])
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv1', trainable=True)(us2)
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(x7)

    # Upsampling 3
    us3 = concatenate([UpSampling2D(size=(2, 2))(x7), x2])
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv1', trainable=True)(us3)
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(x8)

    # Upsampling 4
    us4 = concatenate([UpSampling2D(size=(2, 2))(x8), x1])
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv1', trainable=True)(us4)
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv2', trainable=True)(x9)

    #dense_prediction = Conv2D(
        #num_output_classes,
        #(3, 3),
        #padding='same',
        #activation='sigmoid',
        #kernel_initializer='orthogonal',
        #kernel_regularizer=l2(weight_decay),
        #bias_regularizer=l2(weight_decay))(x9)
    #model = Model(inputs=img_input, outputs=dense_prediction)
    #opt = Adam(1e-4)
    x10 = Conv2D(1, (1, 1), activation='sigmoid')(x9)
    model = Model(inputs=[img_input], outputs=[x10])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[dice_coef])
    return model

def get_Att_unet(input_dim, output_dim, num_output_classes):

    img_input = Input(shape=input_dim, name='image') #（32,32,1）
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(x1)
    attention1_1 = LFSAM()(x1)
    ds1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(attention1_1)

    # Block 2
    # input is of size : 16 x 16 x 64
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(x2)
    attention1_2 = LFSAM()(x2)
    ds2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(attention1_2)

    # Block 3
    # input is of size : 8 x 8 x 128
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=True)(ds2)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=True)(x3)
    attention2_1 = HFCAM()(x3)
    ds3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(attention2_1)

    # Block 4
    # input is of size ：4×4×256
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds3)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=True)(x4)
    attention2_2 = HFCAM()(x4)
    ds4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(attention2_2)

    # Block 5
    # input is of size ：2×2×512
    x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=True)(ds4)
    x5 = Conv2D(1024, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=True)(x5)
    attention2_3 = HFCAM()(x5)

    # Upsampling 1
    up1 = UpSampling2D(size=(2, 2))(attention2_3)
    attention3_1 = CCAM(up1, attention2_2)
    us1 = concatenate([attention3_1, up1])
    #us1 = concatenate([UpSampling2D(size=(2, 2))(x5), x4])
    x6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv1', trainable=True)(us1)
    x6 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv2', trainable=True)(x6)

    # Upsampling 2
    up2 = UpSampling2D(size=(2, 2))(x6)
    attention3_2 = CCAM(up2, attention2_1)
    us2 = concatenate([attention3_2, up2])
    #us2 = concatenate([UpSampling2D(size=(2, 2))(x6), x3])
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv1', trainable=True)(us2)
    x7 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(x7)

    # Upsampling 3
    up3 = UpSampling2D(size=(2, 2))(x7)
    attention3_2 = CCAM(up3, attention1_2)
    us3 = concatenate([attention3_2, up3])
    #us3 = concatenate([UpSampling2D(size=(2, 2))(x7), x2])
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv1', trainable=True)(us3)
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(x8)

    # Upsampling 4
    up4 = UpSampling2D(size=(2, 2))(x8)
    attention3_3 = CCAM(up4, attention1_1)
    us4 = concatenate([attention3_3, up4])
    #us4 = concatenate([UpSampling2D(size=(2, 2))(x8), x1])
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv1', trainable=True)(us4)
    x9 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv2', trainable=True)(x9)

    #dense_prediction = Conv2D(
        #num_output_classes,
        #(3, 3),
        #padding='same',
        #activation='sigmoid',
        #kernel_initializer='orthogonal',
        #kernel_regularizer=l2(weight_decay),
        #bias_regularizer=l2(weight_decay))(x9)
    #model = Model(inputs=img_input, outputs=dense_prediction)
    #opt = Adam(1e-4)
    x10 = Conv2D(1, (1, 1), activation='sigmoid')(x9)
    model = Model(inputs=[img_input], outputs=[x10])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[dice_coef])
    return model

def get_HR_unet():

    img_input = Input(shape=(32, 32, 3), name='image') #（32,32,1）
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(conv1_1)
    ds1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(conv1_2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(conv2_1)
    ds2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(conv2_2)

    x = stem_net(ds2)
    #print(x.shape, 'xstem')

    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])

    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])
    x = fuse__final_layer(x)

    conv3_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)

    us3 = concatenate([UpSampling2D(size=(2, 2))(conv3_1), conv2_2])
    conv4_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv1', trainable=True)(us3)
    conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(conv4_1)

    us4 = concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv1_2])
    conv5_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv1', trainable=True)(us4)
    conv5_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv2', trainable=True)(conv5_1)

    #dense_prediction = Conv2D(
        #num_output_classes,
        #(3, 3),
        #padding='same',
        #activation='sigmoid',
        #kernel_initializer='orthogonal',
        #kernel_regularizer=l2(weight_decay),
        #bias_regularizer=l2(weight_decay))(x9)
    #model = Model(inputs=img_input, outputs=dense_prediction)
    #opt = Adam(1e-4)
    conv6_1 = Conv2D(1, (1, 1), activation='sigmoid')(conv5_2)
    model = Model(inputs=[img_input], outputs=[conv6_1])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[dice_coef])
    return model

def get_HR_Att_unet():

    img_input = Input(shape=(32, 32, 3), name='image') #（32,32,1）

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(conv1_1)
    attention1_1 = LFSAM()(conv1_2)
    ds1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(attention1_1)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(conv2_1)
    attention1_2 = LFSAM()(conv2_2)
    ds2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(attention1_2)

    x = stem_net(ds2)
    #print(x.shape, 'xstem')

    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])

    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])
    x = fuse__final_layer(x)

    conv3_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
    conv3_2 = Conv2D(256, (1, 1), activation='relu', padding='same', name='block3_conv2', trainable=True)(conv3_1)
    attention2_1 = HFCAM()(conv3_2)
    ds3 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(attention2_1)

    conv4_1 = Conv2D(512, (1, 1), activation='relu', padding='same', name='block4_conv1', trainable=True)(ds3)
    conv4_2 = Conv2D(512, (1, 1), activation='relu', padding='same', name='block4_conv2', trainable=True)(conv4_1)
    attention2_2 = HFCAM()(conv4_2)
    ds4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(attention2_2)

    conv5_1 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='block5_conv1', trainable=True)(ds4)
    conv5_2 = Conv2D(1024, (1, 1), activation='relu', padding='same', name='block5_conv2', trainable=True)(conv5_1)
    attention2_3 = HFCAM()(conv5_2)

    up1 = UpSampling2D(size=(2, 2))(attention2_3)
    attention3_1 = CCAM(up1, attention2_2)
    us1 = concatenate([attention3_1, up1])
    #us3 = concatenate([UpSampling2D(size=(2, 2))(conv3_1), conv2_2])
    conv6_1 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv1', trainable=True)(us1)
    conv6_2 = Conv2D(512, (3, 3), activation='relu', padding='same', name='us1_conv2', trainable=True)(conv6_1)

    up2 = UpSampling2D(size=(2, 2))(conv6_2)
    attention3_2 = CCAM(up2, attention2_1)
    us2 = concatenate([attention3_2, up2])
    #us4 = concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv1_2])
    conv7_1 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv1', trainable=True)(us2)
    conv7_2 = Conv2D(256, (3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(conv7_1)

    up3 = UpSampling2D(size=(2, 2))(conv7_2)
    attention3_3 = CCAM(up3, attention1_2)
    us3 = concatenate([attention3_3, up3])
    # us4 = concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv1_2])
    conv8_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv1', trainable=True)(us3)
    conv8_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us3_conv2', trainable=True)(conv8_1)

    up4 = UpSampling2D(size=(2, 2))(conv8_2)
    attention3_4 = CCAM(up4, attention1_1)
    us4 = concatenate([attention3_4, up4])
    # us4 = concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv1_2])
    conv9_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv1', trainable=True)(us4)
    conv9_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us4_conv2', trainable=True)(conv9_1)

    dense_prediction = Conv2D(
        1,
        (3, 3),
        padding='same',
        activation='sigmoid',
        kernel_initializer='orthogonal',
        kernel_regularizer=l2(weight_decay),
        bias_regularizer=l2(weight_decay))(conv9_2)
    model = Model(inputs=img_input, outputs=dense_prediction)
    #opt = Adam(1e-4)
    #conv6_1 = Conv2D(1, (1, 1), activation='sigmoid')(conv5_2)
    #model = Model(inputs=[img_input], outputs=[conv6_1])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[dice_coef])
    return model

def get_HR_Att_unet_1():

    img_input = Input(shape=(32, 32, 1), name='image') #（32,32,1）

    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=True)(img_input)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=True)(conv1_1)
    bn1 = BatchNormalization(axis=3)(conv1_2)
    attention1_1 = LFSAM()(bn1)
    bn2 = BatchNormalization(axis=3)(attention1_1)
    bn2 = Dropout(0.2)(bn2)
    ds1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(bn2)

    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=True)(ds1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=True)(conv2_1)
    bn3 = BatchNormalization(axis=3)(conv2_2)
    attention1_2 = LFSAM()(bn3)
    bn4 = BatchNormalization(axis=3)(attention1_2)
    ds2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(bn4)

    x = stem_net(ds2)
    #print(x.shape, 'xstem')

    x = transition_layer1(x)
    x0 = make_branch1_0(x[0])
    x1 = make_branch1_1(x[1])
    x = fuse_layer1([x0, x1])

    x = transition_layer2(x)
    x0 = make_branch2_0(x[0])
    x1 = make_branch2_1(x[1])
    x2 = make_branch2_2(x[2])
    x = fuse_layer2([x0, x1, x2])
    x = fuse__final_layer(x)

    conv3_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name='block3_conv1', trainable=True)(x)
    bn5 = BatchNormalization(axis=3)(conv3_1)
    attention2 = HFCAM()(bn5)
    bn6 = BatchNormalization(axis=3)(attention2)


    up1 = UpSampling2D(size=(2, 2))(bn6)
    attention3_1 = CCAM(up1, attention1_2)
    us1 = concatenate([attention3_1, up1])
    #us3 = concatenate([UpSampling2D(size=(2, 2))(conv3_1), conv2_2])
    conv4_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us1_conv1', trainable=True)(us1)
    conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='us1_conv2', trainable=True)(conv4_1)
    bn7 = BatchNormalization(axis=3)(conv4_2)

    up2 = UpSampling2D(size=(2, 2))(bn7)
    attention3_2 = CCAM(up2, attention1_1)
    us2 = concatenate([attention3_2, up2])
    #us4 = concatenate([UpSampling2D(size=(2, 2))(conv4_2), conv1_2])
    conv5_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us2_conv1', trainable=True)(us2)
    conv5_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='us2_conv2', trainable=True)(conv5_1)
    bn8 = BatchNormalization(axis=3)(conv5_2)

    #dense_prediction = Conv2D(
        #num_output_classes,
        #(3, 3),
        #padding='same',
        #activation='sigmoid',
        #kernel_initializer='orthogonal',
        #kernel_regularizer=l2(weight_decay),
        #bias_regularizer=l2(weight_decay))(x9)
    #model = Model(inputs=img_input, outputs=dense_prediction)
    #opt = Adam(1e-4)
    conv6_1 = Conv2D(1, (1, 1), activation='sigmoid')(bn8)
    model = Model(inputs=[img_input], outputs=[conv6_1])
    #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=[dice_coef])
    return model