import pandas as pd
import numpy as np

from skimage import io

from skimage.color import rgb2gray

import torch
from torch import nn, optim, utils, hub
import torchvision

import argparse
import time

from matplotlib import pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import os
import cv2

# 導入 LRP
import imp
LRP = imp.load_source('LRP', '__init__.py')
from LRP import lrp

# 特製 model
from model import ResNet50MothClassifierHMC

# standard scaler, 避免整數溢位調整過計算順序
def some_rescale (arr):
    arr_max = arr.max()
    shrinked_std = (arr / arr_max).std()
    divider = shrinked_std * arr_max
    rescaled = (arr - arr.mean()) / divider
    return rescaled

# linear scaler
def linear_rescale (arr, rmin=0, rmax=1):
    return ((arr - arr.min()) / (arr.max() - arr.min())) * (rmax - rmin) + rmin

def load_model(model, pretrained):
    weights = torch.load(pretrained, map_location='cuda:0')
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict) 
    model.load_state_dict(model_dict)

# 計算 LRP 後畫 heatmap, 值有點極端, 稍微做了標準化處理
def get_rel_map(lrp_ruled, input_image, R=None):
    relevance = lrp_ruled.relprop(input_image, R)

    relmap = relevance.clone().detach().squeeze().permute(1,2,0).data.cpu().numpy()

    relmap_rescaled_r3 = relmap[:,:,0]
    relmap_rescaled_g3 = relmap[:,:,1]
    relmap_rescaled_b3 = relmap[:,:,2]

    relmap_rescaled_rgb = np.array([relmap_rescaled_r3, relmap_rescaled_g3, relmap_rescaled_b3]).transpose(1,2,0)
    del relevance
    margin = 0.1
    relmap_rescaled_rgb_adjusted = linear_rescale((some_rescale(relmap_rescaled_rgb) / 1).clip(0, margin), 0, 1)
    mask_adjusted = linear_rescale((some_rescale(relmap_rescaled_rgb) / 1).clip(0, margin), 0.1, 1)
    mask_adjusted_grayscale = rgb2gray(mask_adjusted)
    
    return relmap_rescaled_rgb_adjusted, mask_adjusted_grayscale

# Axis 1 方向的拼圖
def concat_images_1 (imgs, padding_width=2):
    num_of_images_per_row = len(imgs)
    padding_v = np.ones([imgs[0].shape[0], padding_width, 3])
    row_ = imgs[0]
    for idx in range(1, num_of_images_per_row-1):
        row_ = np.concatenate([row_, padding_v, imgs[idx]], axis=1)
    if num_of_images_per_row <= 1:
        return row_
    else:
        return np.concatenate([row_, padding_v, imgs[-1]], axis=1)

# Axis 0 方向的拼圖
def concat_images_0 (imgs, padding_width=2):
    num_of_images_per_col = len(imgs)
    print(num_of_images_per_col)
    padding_h = np.ones([padding_width, imgs[0].shape[1], 3])
    col_ = imgs[0]
    for idx in range(1, num_of_images_per_col-1):
        col_ = np.concatenate([col_, padding_h, imgs[idx]], axis=0)
    if num_of_images_per_col <= 1:
        return col_
    else:
        return np.concatenate([col_, padding_h, imgs[-1]], axis=0)


pretrained = "./resnet50_moth_classifier_hmc_20200806e158.pth"

dataset_path = 'sp_meta.csv'
df = pd.read_csv(dataset_path, sep="\t")

# 移除未鑑定到種的資料
df = df[~df.Species.isna()].reset_index(drop=True)

genus = [s.split(' ')[0] for s in df.Species]
df['Genus'] = genus

# 造出物種清單，並產生每筆資料對應的物種 id, 視為 classification 用的 target y
species_list, species_id = np.unique(df.Species, return_inverse=True)
family_list, family_id = np.unique(df.Family, return_inverse=True)
genus_list, genus_id = np.unique(df.Genus, return_inverse=True)

df['sid'] = species_id
df['fid'] = family_id
df['gid'] = genus_id


# init model, 把模型搬進 GPU:0 的記憶體中
# len(species_list)
model = ResNet50MothClassifierHMC(num_of_species=len(species_list), num_of_genus=len(genus_list), num_of_families=len(family_list)).to('cuda:0')
load_model(model, pretrained)
model = model.eval()

# initializing LRP, 會把 model 中的 backprop 元件都替換成對應的 lrp rules
lrp_ruled = lrp.LRP_hmc(model, 'z_plus')

results = pd.read_csv('valid_and_test_moth_classification_hmc_20200806e158.csv', sep='\t')

# 把辨識對的跟錯的分兩堆
bad_results = results[results.true_sid != results.pred_sid]
bad_results_sp = np.unique(bad_results.true_sname.values)

good_results = results[results.true_sid == results.pred_sid]
good_results_sp = np.unique(good_results.true_sname.values)

dataroot = "/Users/gsmai/Documents/GitHub/vae/save/downloaded256/"

valid_and_test_sp = np.unique(results.true_sname.values)

# 移除未鑑定到種的資料
df_meta_sp = df.Species

# 挑物種看圖
# sp = 'Areas galactina formosana'
# sp = 'Neocerura liturata liturata'
# sp = 'Kamalia tattakana'
# sp = 'Acronicta albistigma'
# sp = 'Aethalura lushanalis'
# sp = 'Barsine aberrans'
#sp = 'Barsine callorufa'

sp = 'Barsine connexa'
sp_N = df_meta_sp[df_meta_sp == sp].shape[0]

sp_all_files = (df[df.Species == sp].Number + '.jpg').values
trained_files = [f for f in sp_all_files if f not in results[results.true_sname==sp].file.values]

trained_images = []
for trained_file in trained_files:
    trained_images.append(io.imread(dataroot + trained_file))

trained_image_mean = np.array(trained_images).mean(axis=0) / 255.

imgs_good_for_lrp = good_results[good_results.true_sname==sp].file.values
imgs_bad_for_lrp = bad_results[bad_results.true_sname==sp].file.values

sp_bad = np.unique(bad_results[bad_results.true_sname==sp].pred_sname)
sp_bad_names = '|'.join(sp_bad)

len_imgs_good_for_lrp = len(imgs_good_for_lrp)
len_imgs_bad_for_lrp = len(imgs_bad_for_lrp)

relmaps_good_sum = np.zeros([256,256,3], dtype=float)
masks_good_sum = np.zeros([256,256,3], dtype=float) + 0.1
images_good_sum = np.zeros([256,256,3], dtype=float)

for img_file_good in imgs_good_for_lrp:
    image_good = io.imread(dataroot + img_file_good)

    input_image_good = torch.tensor(image_good).permute(2,0,1).unsqueeze(0).float() / 255
    input_image_good_cuda = input_image_good.cuda()
    relmap_good_rescaled_rgb, mask_good = get_rel_map(lrp_ruled, input_image_good_cuda)

    images_good_sum += image_good
    relmaps_good_sum += relmap_good_rescaled_rgb
    masks_good_sum += np.array([mask_good, mask_good, mask_good]).transpose(1,2,0)

if len_imgs_good_for_lrp == 0:
    image_good_mean = images_good_sum
    relmap_good_mean = relmaps_good_sum
    mask_good_mean = masks_good_sum
else:
    image_good_mean = images_good_sum / len_imgs_good_for_lrp
    relmap_good_mean = relmaps_good_sum / len_imgs_good_for_lrp
    mask_good_mean = masks_good_sum / len_imgs_good_for_lrp

###############################
relmaps_bad_sum = np.zeros([256,256,3], dtype=float)
masks_bad_sum = np.zeros([256,256,3], dtype=float) + 0.1
images_bad_sum = np.zeros([256,256,3], dtype=float)

for img_file_bad in imgs_bad_for_lrp:
    image_bad = io.imread(dataroot + img_file_bad)

    input_image_bad = torch.tensor(image_bad).permute(2,0,1).unsqueeze(0).float() / 255
    input_image_bad_cuda = input_image_bad.cuda()
    relmap_bad_rescaled_rgb, mask_bad = get_rel_map(lrp_ruled, input_image_bad_cuda)

    images_bad_sum += image_bad
    relmaps_bad_sum += relmap_bad_rescaled_rgb
    masks_bad_sum += np.array([mask_bad, mask_bad, mask_bad]).transpose(1,2,0)

if len_imgs_bad_for_lrp == 0:
    image_bad_mean = images_bad_sum
    relmap_bad_mean = relmaps_bad_sum
    mask_bad_mean = masks_bad_sum
else:
    image_bad_mean = images_bad_sum / len_imgs_bad_for_lrp
    relmap_bad_mean = relmaps_bad_sum / len_imgs_bad_for_lrp
    mask_bad_mean = masks_bad_sum / len_imgs_bad_for_lrp

row_good = concat_images_1([image_good_mean / 255, relmap_good_mean, mask_good_mean * image_good_mean / 255.])
row_bad = concat_images_1([image_bad_mean / 255, relmap_bad_mean, mask_bad_mean * image_bad_mean / 255.])
row_trained = concat_images_1([trained_image_mean, np.zeros_like(trained_image_mean), np.zeros_like(trained_image_mean)])
plt.imshow(concat_images_0([row_good, row_bad, row_trained]))
predicted_n = len_imgs_good_for_lrp + len_imgs_bad_for_lrp
pred_acc = len_imgs_good_for_lrp / predicted_n
trained_n = sp_N - predicted_n

plt.title("%s, acc: %.2f (%d/%d/%d)\n%s" % (sp, pred_acc, len_imgs_good_for_lrp, predicted_n, trained_n, sp_bad_names))
plt.show()
