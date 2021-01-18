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
            input_l = tf.image.rot90(input_l,k=1)  # 90
            input_r = tf.image.rot90(input_r,k=1)  # 90
        else:
            input_l = tf.image.rot90(input_l,k=3)  # 270
            input_r = tf.image.rot90(input_r,k=3)  # 270

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
