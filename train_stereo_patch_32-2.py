import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import numpy as np

from sklearn import metrics
from scipy import interpolate

import tensorflow as tf
import tensorflow_addons as tfa
import math
import cv2 

# Import processing 
from processing import *
from model import *

# Cell 2: load images using Tensorflow

PATH = './patches_for_training/rgbnir_SM_32'

BUFFER_SIZE = 1024 * 4
BATCH_SIZE = 32  # for each positive and negative pairs, altogether = 32
n_train_samples = 117466


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 4

    rgb_pos = image[:, :w, :]
    nir_pos = image[:, w * 1:w * 2, :]
    rgb_neg = image[:, w * 2:w * 3, :]
    nir_neg = image[:, w * 3:w * 4, :]

    rgb_pos = tf.cast(rgb_pos, tf.float32)
    nir_pos = tf.cast(nir_pos, tf.float32)
    rgb_neg = tf.cast(rgb_neg, tf.float32)
    nir_neg = tf.cast(nir_neg, tf.float32)

    return rgb_pos, nir_pos, rgb_neg, nir_neg


# cell 3: data augmentation

def resize(input_l, input_r, target_l, target_r, height, width):
    input_l = tf.image.resize(input_l, [height, width],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_r = tf.image.resize(input_r, [height, width],
                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_l = tf.image.resize(target_l, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    target_r = tf.image.resize(target_r, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_l, input_r, target_l, target_r


def random_crop(input_l, input_r, target_l, target_r):
    stacked_image = tf.stack([input_l, input_r, target_l, target_r], axis=0)
    cropped_image = tf.image.random_crop(stacked_image, size=[4, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1], cropped_image[2], cropped_image[3]


# normalizing the images to [-1, 1]
def normalize(input_l, input_r, target_l, target_r):
    input_l = (input_l / 127.5) - 1
    input_r = (input_r / 127.5) - 1
    target_l = (target_l / 127.5) - 1
    target_r = (target_r / 127.5) - 1

    return input_l, input_r, target_l, target_r


def random_jitter(input_l, input_r, target_l, target_r):
    # resize to 68x68
    #input_l, input_r, target_l, target_r = resize(input_l, input_r, target_l, target_r, 68, 68)

    # crop
    #input_l, input_r, target_l, target_r = random_crop(input_l, input_r, target_l, target_r)

    # flip_left_right
    if tf.random.uniform(()) > 0.5:
        input_l = tf.image.flip_left_right(input_l)
        input_r = tf.image.flip_left_right(input_r)
        target_l = tf.image.flip_left_right(target_l)
        target_r = tf.image.flip_left_right(target_r)

    # flip_up_down
    if tf.random.uniform(()) > 0.5:
        input_l = tf.image.flip_up_down(input_l)
        input_r = tf.image.flip_up_down(input_r)
        target_l = tf.image.flip_up_down(target_l)
        target_r = tf.image.flip_up_down(target_r)

    # brighness change
    if tf.random.uniform(()) > 0.5:
        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        input_l = input_l + rand_value

        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        input_r = input_r + rand_value

        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        target_l = target_l + rand_value

        rand_value = tf.random.uniform((), minval=-5.0, maxval=5.0)
        target_r = target_r + rand_value

    # contrast change
    if tf.random.uniform(()) > 0.5:
        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(input_l)
        input_l = (input_l - mean_value) * rand_value + mean_value

        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(input_r)
        input_r = (input_r - mean_value) * rand_value + mean_value

        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(target_l)
        target_l = (target_l - mean_value) * rand_value + mean_value

        rand_value = tf.random.uniform((), minval=0.8, maxval=1.2)
        mean_value = tf.reduce_mean(target_r)
        target_r = (target_r - mean_value) * rand_value + mean_value

    # clip value
    input_l = tf.clip_by_value(input_l, clip_value_min=0.0, clip_value_max=255.0)
    input_r = tf.clip_by_value(input_r, clip_value_min=0.0, clip_value_max=255.0)
    target_l = tf.clip_by_value(target_l, clip_value_min=0.0, clip_value_max=255.0)
    target_r = tf.clip_by_value(target_r, clip_value_min=0.0, clip_value_max=255.0)

    # rotate positive samples for making hard positive cases
    if tf.random.uniform(()) > 0.5:
        if tf.random.uniform(()) < 0.5:
            input_l = tfa.image.rotate(input_l, 1.5707963268)  # 90
            input_r = tfa.image.rotate(input_r, 1.570796326)  # 90
        else:
            input_l = tfa.image.rotate(input_l, 4.7123889804)  # 270
            input_r = tfa.image.rotate(input_r, 4.7123889804)  # 270

    return input_l, input_r, target_l, target_r


def load_image_train(image_file):
    input_l, input_r, target_l, target_r = load(image_file)
    input_l, input_r, target_l, target_r = random_jitter(input_l, input_r, target_l, target_r)
    input_l, input_r, target_l, target_r = normalize(input_l, input_r, target_l, target_r)

    return input_l, input_r, target_l, target_r


def load_image_test(image_file):
    input_l, input_r, target_l, target_r = load(image_file)
    input_l, input_r, target_l, target_r = resize(input_l, input_r, target_l, target_r, IMG_HEIGHT, IMG_WIDTH)
    input_l, input_r, target_l, target_r = normalize(input_l, input_r, target_l, target_r)

    return input_l, input_r, target_l, target_r

# cell 4: load training data

# train_dataset
train_dataset = tf.data.Dataset.list_files(PATH+'/*.png')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

# cell 5: Network building blocks

def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.keras.initializers.he_normal(seed=None)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    return result

def upsample(filters, size):
    initializer = tf.keras.initializers.he_normal(seed=None)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                    kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())
    result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    return result

def extract_first_features(filters, size, strides, apply_batchnorm=True):
    initializer = tf.keras.initializers.he_normal(seed=None)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                             kernel_initializer=initializer, use_bias=False,
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
        result.add(tfa.layers.InstanceNormalization())

    result.add(tf.keras.layers.ReLU())

    return result


# cell 6: RGB2NIR network

def RGB2NIR_convertor(input_x):
    # input shape:  (64, 64, 3)
    # output shape: (64, 64, 1)

    x_1 = input_x

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 32, 32, 64)
        downsample(128, 4, apply_batchnorm=True),  # (bs, 16, 16, 512)
        downsample(256, 4, apply_batchnorm=True),  # (bs, 8, 8, 512)
        downsample(256, 4, apply_batchnorm=True),  # (bs, 4, 4, 512)
        downsample(256, 4, apply_batchnorm=True),  # (bs, 2, 2, 512)
        #downsample(256, 4, apply_batchnorm=True),  # (bs, 1, 1, 512)
    ]

    up_stack = [
        #upsample(256, 4),  # (bs, 2, 2, 1024)
        upsample(256, 4),  # (bs, 4, 4, 1024)
        upsample(256, 4),  # (bs, 8, 8, 1024)
        upsample(128, 4),  # (bs, 16, 16, 1024)
        upsample(64, 4),  # (bs, 32, 32, 512)
    ]

    initializer = tf.keras.initializers.he_normal(seed=None)
    OUTPUT_CHANNELS = 1
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 64, 64, 1)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x_1 = down(x_1)
        skips.append(x_1)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    concat = tf.keras.layers.Concatenate()
    for up, skip in zip(up_stack, skips):
        x_1 = up(x_1)
        x_1 = concat([x_1, skip])

    x_1 = last(x_1)

    return x_1


# cell 7: NIR2RGB network

def NIR2RGB_convertor(input_x):
    # input shape:  (64, 64, 1)
    # output shape: (64, 64, 3)

    x_1 = input_x

    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (bs, 32, 32, 64)
        downsample(128, 4, apply_batchnorm=True),  # (bs, 16, 16, 128)
        downsample(256, 4, apply_batchnorm=True),  # (bs, 8, 8, 256)
        downsample(256, 4, apply_batchnorm=True),  # (bs, 4, 4, 256)
        downsample(256, 4, apply_batchnorm=True),  # (bs, 2, 2, 256)
        #downsample(256, 4, apply_batchnorm=True),  # (bs, 1, 1, 256)
    ]

    up_stack = [
        #upsample(256, 4),  # (bs, 2, 2, 256)
        upsample(256, 4),  # (bs, 4, 4, 256)
        upsample(256, 4),  # (bs, 8, 8, 256)
        upsample(128, 4),  # (bs, 16, 16, 128)
        upsample(64, 4),  # (bs, 32, 32, 64)
    ]

    initializer = tf.keras.initializers.he_normal(seed=None)
    OUTPUT_CHANNELS = 3
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 64, 64, 1)

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x_1 = down(x_1)
        skips.append(x_1)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    concat = tf.keras.layers.Concatenate()
    for up, skip in zip(up_stack, skips):
        x_1 = up(x_1)
        x_1 = concat([x_1, skip])

    x_1 = last(x_1)

    return x_1


# cell 8: NIR domain matching

def NIR_domain_matching(input_x1, input_x2):
    x_1 = input_x1
    x_2 = input_x2

    # for x_1
    layer1 = extract_first_features(32, 3, 1, True)
    layer2 = extract_first_features(64, 3, 1, True)
    layer3 = extract_first_features(128, 3, 1, True)
    layer4 = extract_first_features(128, 5, 2, True)
    layer5 = extract_first_features(256, 3, 1, True)
    layer6 = extract_first_features(256, 5, 2, True)
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

    return x


# cell 9: RGB domain matching

def RGB_domain_matching(input_x1, input_x2):
    x_1 = input_x1
    x_2 = input_x2

    # for x_1
    layer1 = extract_first_features(32, 3, 1, True)
    layer2 = extract_first_features(64, 3, 1, True)
    layer3 = extract_first_features(128, 3, 1, True)
    layer4 = extract_first_features(128, 5, 2, True)
    layer5 = extract_first_features(256, 3, 1, True)
    layer6 = extract_first_features(256, 5, 2, True)
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

    return x


# cell 10: construct SPIMNet network

def make_similarity_model():
    inputs_1 = layers.Input(shape=[32, 32, 3])
    inputs_2 = layers.Input(shape=[32, 32, 1])
    x_rgb = inputs_1
    x_nir = inputs_2

    # convert domains
    x_converted_nir = RGB2NIR_convertor(x_rgb)
    x_converted_rgb = NIR2RGB_convertor(x_nir)

    # matching
    f_nir = NIR_domain_matching(x_nir, x_converted_nir)
    f_rgb = RGB_domain_matching(x_rgb, x_converted_rgb)

    # concat features
    x = tf.concat([f_nir, f_rgb], 1)

    # metric learning
    x = layers.Dense(1024)(x)
    x = layers.Dense(128)(x)
    x = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=[x, x_rgb, x_converted_rgb, x_nir, x_converted_nir])
    return model

similaritor = make_similarity_model()
#similaritor.summary()

# cell 12: Construct content loss (~perceptual loss)

from tensorflow.python.keras.applications.vgg19 import VGG19


def vgg_54():
    return _vgg(20)


def _vgg(output_layer):
    vgg = VGG19(input_shape=(32, 32, 3), include_top=False, weights='imagenet')
    return tf.keras.Model(vgg.input, vgg.layers[output_layer].output)


vgg = vgg_54()

mean_squared_error = tf.keras.losses.MeanSquaredError()


def content_loss(img1, img2):
    #print("SHAPE",img1.shape)
    # img1= tf.image.resize(img1,(32,32))
    # img2= tf.image.resize(img2,(32,32))

    img1_fea = vgg(img1)
    img2_fea = vgg(img2)

    loss = mean_squared_error(img1_fea, img2_fea)

    return loss


# cell 13: build loss function

# path to save checkpoints
checkpoint_dir = './checkpoint/32x32_oldmodel'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(step=tf.Variable(1),similaritor=similaritor)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)

# Instantiate an optimizer.
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.01)

beta = 30.0
alpha = 0.1


def similaritor_loss(pos_output, x_rgb, x_converted_rgb, x_nir, x_converted_nir, neg_output):
    # total_loss1
    pos_loss = cross_entropy(tf.ones_like(pos_output), pos_output)
    neg_loss = cross_entropy(tf.zeros_like(neg_output), neg_output)
    total_loss1 = pos_loss + neg_loss

    # total_loss3
    l1_loss1 = tf.reduce_mean(tf.abs(x_rgb - x_converted_rgb))
    l1_loss2 = tf.reduce_mean(tf.abs(x_nir - x_converted_nir))
    total_loss3 = alpha * l1_loss1 + alpha * l1_loss2

    # total_loss2
    pos_nir1 = tf.concat([x_nir, x_nir, x_nir], 3)
    pos_nir2 = tf.concat([x_converted_nir, x_converted_nir, x_converted_nir], 3)

    pos_nir_loss = content_loss(pos_nir1, pos_nir2)
    pos_rgb_loss = content_loss(x_rgb, x_converted_rgb)
    total_loss2 = pos_nir_loss * beta + pos_rgb_loss * beta

    # total_loss
    total_loss = total_loss1 + total_loss2 + total_loss3
    return total_loss, total_loss1, total_loss2, total_loss3


# cell 14: train SPIMNet

def train(train_data):
    for epoch in range(1, 15):
        print("EPOCH: ", epoch)
        # learning rate
        if epoch < 20:
            lr = 1e-3
        else:
            lr = 1e-4
        optimizer = tf.keras.optimizers.Adam(lr)

        average_loss = 0
        average_posl = 0
        average_negl = 0
        average_l1lo = 0

        count = 0
        count_ones_pos = 0
        count_ones_neg = 0

        iter_samples = math.ceil(n_train_samples/BATCH_SIZE)
        progbar = tf.keras.utils.Progbar(iter_samples)

        for pos_bs_img0, pos_bs_img1, neg_bs_img0, neg_bs_img1 in train_data:
            progbar.update(count+1) # This will update the progress bar graph.
            pos_bs_img1 = pos_bs_img1[:, :, :, 0:1]
            neg_bs_img1 = neg_bs_img1[:, :, :, 0:1]

            with tf.GradientTape() as sim_tape:
                # training
                pos_output, x_rgb, x_c_rgb, x_nir, x_c_nir = similaritor([pos_bs_img0, pos_bs_img1], training=True)
                neg_output, _, _, _, _ = similaritor([neg_bs_img0, neg_bs_img1], training=True)

                sim_loss, pos_loss, neg_loss, total_loss3 = similaritor_loss(pos_output, x_rgb, x_c_rgb, x_nir, x_c_nir,
                                                                             neg_output)

                # --------- compute training acc ---------
                bool_pos_output = pos_output > 0
                ones_pos_output = tf.reduce_sum(tf.cast(bool_pos_output, tf.float32))
                count_ones_pos = count_ones_pos + ones_pos_output

                bool_neg_output = neg_output < 0
                ones_neg_output = tf.reduce_sum(tf.cast(bool_neg_output, tf.float32))
                count_ones_neg = count_ones_neg + ones_neg_output

            gradients = sim_tape.gradient(sim_loss, similaritor.trainable_variables)
            optimizer.apply_gradients(zip(gradients, similaritor.trainable_variables))
            
            checkpoint.step.assign_add(1)
            
            if ( count % 100 == 0):
            #   save_path = manager.save()
            #   print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
              print('\t Loss at step: %d: Normal loss: %.6f, Negative Loss: %.6f, Sim loss: %.6f' % (int(checkpoint.step), pos_loss,neg_loss,total_loss3))

            average_loss = average_loss + sim_loss
            average_posl = average_posl + pos_loss
            average_negl = average_negl + neg_loss
            average_l1lo = average_l1lo + total_loss3

            count = count + 1

        average_loss = average_loss / count
        average_posl = average_posl / count
        average_negl = average_negl / count
        average_l1lo = average_l1lo / count

        print('epoch {}  average_loss {}  lr {}'.format(epoch, average_loss, lr))
        print('normal loss {}  perceptual loss {}  l1 loss {}'.format(average_posl, average_negl, average_l1lo))

        pos_acc = (count_ones_pos * 100.0) / n_train_samples
        neg_acc = (count_ones_neg * 100.0) / n_train_samples
        print('train acc (pos) {} - acc (neg) {}'.format(pos_acc, neg_acc))

        save_path = manager.save()
        print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
        print('')

# cell 15: train SPIMNet with 35 epochs
checkpoint.restore(manager.latest_checkpoint)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
if manager.latest_checkpoint:
  print("Restored from {}".format(manager.latest_checkpoint))
else:
  print("Initializing from scratch.")
train(train_dataset)
