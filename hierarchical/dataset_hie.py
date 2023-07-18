import torch
import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageOps
#import random
import numpy as np
import torchvision.transforms as transforms

import skimage.io as io
from imgaug import augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
most_of_the_time = lambda aug: iaa.Sometimes(0.9, aug)
usually = lambda aug: iaa.Sometimes(0.75, aug)
always = lambda aug: iaa.Sometimes(1, aug)
charm = lambda aug: iaa.Sometimes(0.33, aug)
seldom = lambda aug: iaa.Sometimes(0.2, aug)

# image augmentation, 可以參考 imgaug 的文件
# https://github.com/aleju/imgaug
augseq_special = iaa.Sequential([
    iaa.Fliplr(0.5)
    ,sometimes(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
            rotate=(-30, 30), # rotate by -45 to +45 degrees
            mode='symmetric',
        ))
])

# 這邊是配合 torch utils 的 dataset loader 的寫法，基本上就是依樣畫葫蘆
class ImageDatasetFromFileSpecial(data.Dataset):
    def __init__(self, image_list, root_path, y=None, aug=True, return_filename=False):
        super(ImageDatasetFromFileSpecial, self).__init__()
                
        self.image_filenames = image_list
        self.y = y
        self.aug = aug
        self.root_path = root_path
        self.return_filename = return_filename
                       
        self.input_transform = transforms.Compose([ 
                                   transforms.ToTensor()                                                                      
                               ])

    def __getitem__(self, index):
          
        img = io.imread(join(self.root_path, self.image_filenames[index]))
        if self.aug:
          img = augseq_special.augment_images([img])
        else:
          img = [img]

        #print(img[0].shape)
        img = self.input_transform(img[0].copy())
        #print(img)
        
        if self.y is None:
          if self.return_filename:
            return img, self.image_filenames[index]
          else:
            return img
        else:
          if self.return_filename:
            return img, self.y[index], self.image_filenames[index]
          else:
            return img, self.y[index]

    def __len__(self):
        return len(self.image_filenames)

class GeneralDataset(data.Dataset):
    def __init__(self, x, y):
        super(GeneralDataset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]