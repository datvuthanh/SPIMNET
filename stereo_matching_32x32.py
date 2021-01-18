import os
import PIL
from PIL import Image
import numpy as np
from pathlib import Path
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_addons as tfa
import sys

def jointbilateral(im, gd):
    std_c = 0.05
    std_d = 10
    size = 7
    radius = size // 2

    paddings = tf.constant([[radius, radius],[radius,radius],[0,0]])

    im_pad = tf.pad(im, paddings, "REFLECT")
    gd_pad = tf.pad(gd, paddings, "REFLECT")

    height = tf.shape(gd_pad)[0]
    width = tf.shape(gd_pad)[1]

    gd_cp = gd_pad[radius:height - radius, radius:width - radius, :]

    smooth_sum = tf.zeros_like(gd_cp)
    wt_sum = tf.zeros_like(gd_cp)
    for y in range(size):
        for x in range(size):
            gd_wt = (gd_pad[y:height - (2 * radius) + y, x:width - (2 * radius) + x, :] - gd_cp) * \
                    (gd_pad[y:height - (2 * radius) + y, x:width - (2 * radius) + x, :] - gd_cp)
            gd_wt = tf.math.exp(-1 * gd_wt / (2 * std_c * std_c))
            gd_wt = tf.math.reduce_mean(gd_wt, axis=2, keepdims=True)
            dist_wt = ((radius - y) * (radius - y) + (radius - x) * (radius - x)) * tf.ones_like(gd_wt)
            dist_wt = tf.math.exp(-1 * dist_wt / (2 * std_d * std_d))
            im_src = im_pad[y:height - (2 * radius) + y, x:width - (2 * radius) + x, :]

            smooth = gd_wt * dist_wt * im_src
            wt = gd_wt * dist_wt
            smooth_sum = smooth_sum + smooth
            wt_sum = wt_sum + wt

    result = smooth_sum / wt_sum
    return result


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


data_path = './test_dataset'
list_path = './lists'
path_save_disp = './predict'
folders = ['20170222_0951', '20170222_1423', '20170223_1639', '20170224_0742']
records = []
maxd = 26
def stereo_matching(model):
    similaritor = model
    checkpoint_dir = './checkpoint/32x32_oldmodel'
    checkpoint = tf.train.Checkpoint(similaritor=similaritor)
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

    for folder in folders:
        f = open(Path(list_path) / (folder + '.txt'), 'r')
        lines = f.readlines()
        f.close()
        for i, line in enumerate(lines):
            splited = line.split()
            collection = splited[0]
            key = splited[1]
            rgb_exp = float(splited[2])
            nir_exp = float(splited[3])
            record = [collection, key, rgb_exp, nir_exp]
            records.append(record)
    # for record in records:
    #     print('')
    #     print(record[1])
    path_load_c_l = './test_dataset/20170222_0951/RGBResize/220951_001873_RGBResize.png'
    path_load_c_r = './test_dataset/20170222_0951/NIRResize/220951_001873_NIRResize.png' #data_path + '/' + record[0] + '/NIRResize/' + record[1] + '_NIRResize.png'

    tmp_clr = PIL.Image.open(path_load_c_l)
    tmp_nir = PIL.Image.open(path_load_c_r)
    tmp_clr = np.asarray(tmp_clr)
    tmp_nir = np.asarray(tmp_nir)
    tmp_clr = tf.convert_to_tensor(tmp_clr, dtype=tf.float32)
    tmp_nir = tf.convert_to_tensor(tmp_nir, dtype=tf.float32)
    tmp_clr = tf.expand_dims(tmp_clr, 0)
    tmp_nir = tf.expand_dims(tmp_nir, 0)
    tmp_nir = tf.expand_dims(tmp_nir, 3)

    # Make a warping coordinates
    flow_x = tf.ones_like(tmp_nir)
    flow_y = tf.zeros_like(tmp_nir)

    # Stack each viewpoint image by warping the image (It is similar to making the batch size. In this case, the maximum disparity range could be the batch size.)
    for d in range(maxd):
        flow_x_d = flow_x * d
        flow = tf.concat([flow_y, flow_x_d], axis=3)
        tmp_nir_d = tfa.image.dense_image_warp(tmp_nir, flow)
        if d == 0:
            batch_clr = tmp_clr
            batch_nir = tmp_nir_d
        else:
            batch_clr = tf.concat([batch_clr, tmp_clr], axis=0)
            batch_nir = tf.concat([batch_nir, tmp_nir_d], axis=0)
    # Padding for image crop (I put pad on the bottom and the right side of image)
    padding = 32
    right_pad_clr = batch_clr[:,:,550:,:]
    bottom_pad_clr = batch_clr[:,397:,:,:]
    bottom_right_clr = batch_clr[:,397:,550:,:]
    bottom_pad_clr = tf.concat([bottom_pad_clr, bottom_right_clr], 2)
    clr_padding = tf.concat([batch_clr, right_pad_clr], 2)
    clr_padding = tf.concat([clr_padding, bottom_pad_clr], 1)

    right_pad_nir = batch_nir[:,:,550:,:]
    bottom_pad_nir = batch_nir[:,397:,:,:]
    bottom_right_nir = batch_nir[:,397:,550:,:]
    bottom_pad_nir = tf.concat([bottom_pad_nir, bottom_right_nir], 2)
    nir_padding = tf.concat([batch_nir, right_pad_nir], 2)
    nir_padding = tf.concat([nir_padding, bottom_pad_nir], 1)

    # Crop the image and do the stereo matching
    cnt = 0
    y_iter = clr_padding.shape[1] - padding
    x_iter = clr_padding.shape[2] - padding
    costs = np.zeros((y_iter,x_iter,maxd))
    
    progbar = tf.keras.utils.Progbar(y_iter*x_iter)

    for y in range(y_iter):
        cost_row = np.zeros((1, x_iter, maxd))
        for x in range(x_iter):
            progbar.update(cnt) # This will update the progress bar graph.
            #printProgress(cnt, y_iter*x_iter, 'Progress:', 'Complete', 1, 50)
            rgb_patch = clr_padding[:, y:y + padding, x:x + padding, :] # Crop the image
            nir_patch = nir_padding[:, y:y + padding, x:x + padding, :] # Crop the image
            rgb_patch = (rgb_patch / 127.5) - 1
            nir_patch = (nir_patch / 127.5) - 1
            score, _, _, _, _ = similaritor([rgb_patch, nir_patch], training=False)
            #score = tf.reshape(score, [1, 1, maxd])
            cost_row[:,x,:] =  tf.squeeze(score) #tf.concat([cost_row, score], 1)
            cnt = cnt + 1
        costs[y,:,:] = cost_row #tf.concat([costs, cost_row], 0)
    
    with open('cost.npy', 'wb') as f:
      np.save(f,costs)
    cost_vol = tf.math.sigmoid(costs) # Make the cost volume

    # for d in range(maxd):
    #     tmp_jbf = jointbilateral(cost_vol[:,:,d:d+1], (tmp_clr[0,:,:,:]/255.0)) # Cost aggregation using bilateral filter
    #     if d==0:
    #         cost_jbf = tmp_jbf
    #     else:
    #         cost_jbf = tf.concat([cost_jbf, tmp_jbf], axis=2)
    # cost_vol_jbf = tf.nn.softmax(cost_jbf, axis=2) # Change the cost to the probability
    disp = tf.math.argmax(cost_vol, axis=2)
    disp = disp.numpy()
    disp = Image.fromarray(disp.astype(np.uint8))
    disp.save('test_32.png')

model = make_similarity_model()
stereo_matching(model)
