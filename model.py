from tensorflow.keras.layers import BatchNormalization as BatchNorm
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Reshape, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import tensorflow as tf


def extract_first_features(filters, size, apply_batchnorm=True):
    initializer = tf.keras.initializers.he_normal(seed=None)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, padding='valid',
                             kernel_initializer=initializer, use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
        #result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


def NIR_domain_matching(input_x1, input_x2):
    x_1 = input_x1
    x_2 = input_x2

    # for x_1
    layer1 = extract_first_features(32, 3, True)
    layer2 = extract_first_features(64, 3, True)
    layer3 = extract_first_features(128, 3, True)
    layer4 = extract_first_features(128, 3, True)
    layer5 = extract_first_features(256, 3, True)
    layer6 = extract_first_features(256, 3, True)
    #layer7 = extract_first_features(256, 3, 1, True)
    #layer8 = extract_first_features(256, 5, 2, True)

    # for x_1
    x_1 = layer1(x_1)
    x_1 = layer2(x_1)
    x_1 = layer3(x_1)
    x_1 = layer4(x_1)
    x_1 = layer5(x_1)
    x_1 = layer6(x_1)
    #x_1 = layer7(x_1)
    #x_1 = layer8(x_1)
    x_1 = layers.Flatten()(x_1)

    # for x_2
    x_2 = layer1(x_2)
    x_2 = layer2(x_2)
    x_2 = layer3(x_2)
    x_2 = layer4(x_2)
    x_2 = layer5(x_2)
    x_2 = layer6(x_2)
    #x_2 = layer7(x_2)
    #x_2 = layer8(x_2)
    x_2 = layers.Flatten()(x_2)

    x = tf.abs(x_1 - x_2)
    #x = tf.concat([x_1, x_2, x], 1)
    #x = tf.reduce_sum(tf.multiply(x_1, x_2), axis=3, name='map_inner_product') # Batch x 1 x 201

    return x


def RGB_domain_matching(input_x1, input_x2):
    x_1 = input_x1
    x_2 = input_x2
    # for x_1
    layer1 = extract_first_features(32, 3, True)
    layer2 = extract_first_features(64, 3, True)
    layer3 = extract_first_features(128, 3, True)
    layer4 = extract_first_features(128, 3, True)
    layer5 = extract_first_features(256, 3, True)
    layer6 = extract_first_features(256, 3, True)
    #layer7 = extract_first_features(256, 3, 1, True)
    #layer8 = extract_first_features(256, 5, 2, True)

    # for x_1
    x_1 = layer1(x_1)
    x_1 = layer2(x_1)
    x_1 = layer3(x_1)
    x_1 = layer4(x_1)
    x_1 = layer5(x_1)
    x_1 = layer6(x_1)
    #x_1 = layer7(x_1)
    #x_1 = layer8(x_1)
    x_1 = layers.Flatten()(x_1)

    # for x_2
    x_2 = layer1(x_2)
    x_2 = layer2(x_2)
    x_2 = layer3(x_2)
    x_2 = layer4(x_2)
    x_2 = layer5(x_2)
    x_2 = layer6(x_2)
    #x_2 = layer7(x_2)
    #x_2 = layer8(x_2)
    x_2 = layers.Flatten()(x_2)

    x = tf.abs(x_1 - x_2)
    #x = tf.concat([x_1, x_2, x], 1)
    #x = tf.reduce_sum(tf.multiply(x_1, x_2), axis=3, name='map_inner_product') # Batch x 1 x 201

    return x

def downsampling(x, level, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, pool_size = 2, pool_strides = 2, regularizer = None, regularizer_param = 0.001,input_name=None):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=conv_strides, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = input_name + 'downsampling_' + str(level) + '_conv_' + str(i))(x)
        if batch_norm:
            x = BatchNorm(name = input_name + 'downsampling_' + str(level) + '_batchnorm_' + str(i))(x)
        x = Activation(activation, name = input_name + 'downsampling_' + str(level) + '_activation_' + str(i))(x)
    skip = x
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    return x, skip

def bottleneck_dilated(x, filters, kernel_size, num_convs = 6, activation = 'relu', batch_norm = False, last_activation = False, regularizer = None, regularizer_param = 0.001,input_name=None):
#     assert num_convs == len(conv_strides)
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    skips = []
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=kernel_size, strides=1, dilation_rate = 2 ** i, activation='relu', padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = input_name + 'bottleneck_skip_' + str(i))(x)
        skips.append(x)
    x = layers.add(skips)
    if last_activation:
        x = Activation('relu')(x)
    return x
    
def bottleneck(x, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, pool_size = 2, pool_strides = 2, regularizer = None, regularizer_param = 0.001,input_name = None):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = input_name + 'bottleneck_' + str(i))(x)
        if batch_norm:
            x = BatchNorm()(x)
        x = Activation(activation)(x)
    return x

def upsampling(x, level, skip, filters, kernel_size, num_convs, conv_strides=1, activation = 'relu', batch_norm = False, conv_transpose = True, upsampling_size = 2, upsampling_strides = 2, regularizer = None, regularizer_param = 0.001,input_name=None):
    if regularizer is not None:
        if regularizer == 'l2':
            reg = regularizers.l2(regularizer_param)
        elif regularizer == 'l1':
            reg = regularizers.l1(regularizer_param)
    else:
        reg = None
    if conv_transpose:
        x = Conv2DTranspose(filters=filters, kernel_size = upsampling_size, strides=upsampling_strides, name = input_name + 'upsampling_' + str(level) + '_conv_trans_' + str(level))(x)
    else:
        x = UpSampling2D((upsampling_size), name = input_name +  'upsampling_' + str(level) + '_ups_' + str(i))(x)
    x = Concatenate()([x, skip])
    for i in range(num_convs):
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', kernel_regularizer=reg, bias_regularizer=reg, name = input_name +  'upsampling_' + str(level) + '_conv_' + str(i))(x)
        if batch_norm:
            x = BatchNorm(name = input_name +  'upsampling_' + str(level) + '_batchnorm_' + str(i))(x)
        x = Activation(activation, name = input_name +  'upsampling_' + str(level) + '_activation_' + str(i))(x)
    return x

def model_simple_unet_initializer(input=None,num_levels = 4, num_layers = 2, num_bottleneck = 2, filter_size_start = 32, kernel_size = 3, bottleneck_dilation = True, bottleneck_sum_activation = False, regularizer = 'l2', regularizer_param = 0.001,output_channels=3,input_name=None):
    inputs = input
    x = inputs
    skips = []
    for i in range(num_levels):
        x, skip = downsampling(x, i, filter_size_start * (2 ** i), kernel_size, num_layers, batch_norm=True, regularizer= regularizer, regularizer_param=regularizer_param,input_name=input_name)
        skips.append(skip)
    if bottleneck_dilation:
        x = bottleneck_dilated(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=True, last_activation=bottleneck_sum_activation, regularizer= regularizer, regularizer_param=regularizer_param,input_name=input_name)
    else:
        x = bottleneck(x, filter_size_start * (2 ** num_levels), kernel_size, num_bottleneck, batch_norm=True, regularizer=regularizer, regularizer_param=regularizer_param,input_name=input_name)
    for j in range(num_levels):      
        last_layer = num_levels - 1
        if j < last_layer:
          x = upsampling(x, j, skips[num_levels - j - 1], filter_size_start * (2 ** (num_levels - j - 1)), kernel_size, num_layers, batch_norm=True, regularizer= regularizer, regularizer_param=regularizer_param,input_name=input_name)
        if j == last_layer:
          x = upsampling(x, j, skips[num_levels - j - 1], output_channels, kernel_size, num_layers, batch_norm=True, regularizer= regularizer, regularizer_param=regularizer_param,input_name=input_name)
    return x