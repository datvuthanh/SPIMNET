## Merge postive and negative pairs
import os
import numpy as np
import PIL
from PIL import Image

path_pos = '/home/memi/Desktop/SPIMNET/patches_for_training/pos_new_10'
path_neg = '/home/memi/Desktop/SPIMNET/patches_for_training/neg_new_10'
path_save = '/home/memi/Desktop/SPIMNET/patches_for_training/rgbnir_SM_10'

print('3. Merge postive and negative pairs')

file_cnt_pos = 0
for pack in os.walk(path_pos):
    for f in pack[2]:
        file_cnt_pos += 1
print('   Number of positive patches: ', file_cnt_pos)

file_cnt_neg = 0
for pack in os.walk(path_neg):
    for f in pack[2]:
        file_cnt_neg += 1
print('   Number of negative patches: ', file_cnt_neg)

merge = []
cnt = 0
for i in range(0, file_cnt_pos):
    pos_patch_path  = '{}/{}.png'.format(path_pos, i)
    neg_patch_path  = '{}/{}.png'.format(path_neg, i)
    pos_patch = PIL.Image.open(pos_patch_path)
    neg_patch = PIL.Image.open(neg_patch_path)
   
    merge = [pos_patch, neg_patch]
    
    imgs_comb = np.hstack((np.asarray(j) for j in merge))
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    save_path = '{}/{}.png'.format(path_save, cnt)
    imgs_comb.save(save_path)
    cnt += 1
    
print('Number of merged patches: ', cnt)