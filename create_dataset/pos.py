import numpy as np
import PIL
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import random

save_path = '/home/memi/Desktop/SPIMNET/patches_for_training/pos_new_10/'
data_path = '/home/memi/Desktop/SPIMNET/dataset'
disp_path = '/home/memi/Desktop/SPIMNET/disparity_maps'
list_path = '/home/memi/Desktop/SPIMNET/lists'
folders = ['20170221_1357', '20170222_0715', '20170222_1207', '20170222_1638', '20170223_0920', '20170223_1217', '20170223_1445', '20170224_1022']
size = 5
records = []
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

cnt = 117373
for record in records:
    path_load_c_l = data_path + '/' + record[0] + '/RGBResize/' + record[1] + '_RGBResize.png'
    path_load_c_r = data_path + '/' + record[0] + '/NIRResize/' + record[1] + '_NIRResize.png'
    path_load_disp_l = disp_path + '/left_pngs/' + record[1]+'.png'
    path_load_disp_r = disp_path + '/right_pngs/' + record[1]+'.png'
    
    tmp_clr = PIL.Image.open(path_load_c_l)
    tmp_nir = PIL.Image.open(path_load_c_r).convert('L')
    
    tmp_disp_l = PIL.Image.open(path_load_disp_l)
    tmp_disp_r = PIL.Image.open(path_load_disp_r)
    
    tmp_clr = np.asarray(tmp_clr)
    tmp_nir = np.asarray(tmp_nir)
    
    tmp_disp_l = np.asarray(tmp_disp_l)
    tmp_disp_r = np.asarray(tmp_disp_r)    

    tmp_nir = tmp_nir[:, :, np.newaxis]
    #tmp_disp_l = np.around((tmp_disp_l[:, :, 0:1] / 255.) * (582. * 0.031)).astype(int)
    #tmp_disp_r = np.around((tmp_disp_r[:, :, 0:1] / 255.) * (582. * 0.031)).astype(int)

    tmp_nir = np.concatenate((tmp_nir, tmp_nir, tmp_nir), 2)
    
    list_x = []
    list_y = []
    
    default_x = random.randint(100, 500)
    default_y = random.randint(100, 300)

    for i in range(3):
        while default_x in list_x:
            default_x = random.randint(100, 500)
        list_x.append(default_x)
        while default_y in list_y:
            default_y = random.randint(100, 300)
        list_y.append(default_y)
    
    left_disp1 = tmp_disp_l[list_y[0], list_x[0], 0]
    right_disp1 = tmp_disp_r[list_y[0], list_x[0] - left_disp1, 0]
    left_disp2 = tmp_disp_l[list_y[1], list_x[1], 0]
    right_disp2 = tmp_disp_r[list_y[1], list_x[1] - left_disp2, 0]
    left_disp3 = tmp_disp_l[list_y[2], list_x[2], 0]
    right_disp3 = tmp_disp_r[list_y[2], list_x[2] - left_disp3, 0]
    
    if left_disp1 == right_disp1:
        left_patch = tmp_clr[list_y[0]-size:list_y[0]+size, list_x[0]-size:list_x[0]+size, :]
        right_patch = tmp_nir[list_y[0]-size:list_y[0]+size, list_x[0]-left_disp1-size:list_x[0]-left_disp1+size, :]
        imgs = [left_patch, right_patch] 
        imgs_comb = np.hstack(imgs[i] for i in range(2))
        imgs_comb = Image.fromarray(imgs_comb)
        path_save  = save_path + str(cnt) + '.png'
        imgs_comb.save(path_save)
        cnt = cnt + 1
    if left_disp2 == right_disp2:
        left_patch = tmp_clr[list_y[1]-size:list_y[1]+size, list_x[1]-size:list_x[1]+size, :]
        right_patch = tmp_nir[list_y[1]-size:list_y[1]+size, list_x[1]-left_disp2-size:list_x[1]-left_disp2+size, :]
        imgs = [left_patch, right_patch] 
        imgs_comb = np.hstack(imgs[i] for i in range(2))
        imgs_comb = Image.fromarray(imgs_comb)
        path_save  = save_path + str(cnt) + '.png'
        imgs_comb.save(path_save)
        cnt = cnt + 1
    if left_disp3 == right_disp3:
        left_patch = tmp_clr[list_y[2]-size:list_y[2]+size, list_x[2]-size:list_x[2]+size, :]
        right_patch = tmp_nir[list_y[2]-size:list_y[2]+size, list_x[2]-left_disp3-size:list_x[2]-left_disp3+size, :]
        imgs = [left_patch, right_patch] 
        imgs_comb = np.hstack(imgs[i] for i in range(2))
        imgs_comb = Image.fromarray(imgs_comb)
        path_save  = save_path + str(cnt) + '.png'
        imgs_comb.save(path_save)
        cnt = cnt + 1
    