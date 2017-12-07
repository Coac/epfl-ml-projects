# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:13:03 2017

@author: santi
"""

import matplotlib.image as mpimg
import matplotlib.patches as matplotlib_patches
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import timeit
from PIL import Image
from functions_v1 import *
from scipy import ndimage as ndi
from skimage import feature

os.chdir('C:/Users/santi/Desktop/ICAI/Master/Segundo/Machine Learning/Labs/epfl-ml-projects/epfl-ml-project2')
wd = os.getcwd()

print(wd)


#### Lad Data
# Loaded a set of images
root_dir = "datas/training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
files.sort()
n = min(100, len(files)) # Load maximum 20 images
print("Loading " + str(n) + " images")
imgs = [load_image(image_dir + files[i]) for i in range(n)]
print(files[0])

gt_dir = root_dir + "groundtruth/"
print("Loading " + str(n) + " groundtruth")
gt_imgs = [load_image(gt_dir + files[i]) for i in range(n)]
print(files[0])

n = 100 # Only use 10 images for training

#### Get edges
for i in range(10):
    im = load_image_pil(image_dir + files[i], bw=True)
    im = np.array(im)
    
    edges = feature.canny(im, sigma = 1.8)
    fill = ndi.binary_fill_holes(edges)
    
    fig1 = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='Greys_r');
    plt.subplot(1, 2, 2)
    plt.imshow(fill, cmap='Greys_r');

## Region based (Elevation map)
from skimage.filters import scharr

for i in range(10):
    im = load_image_pil(image_dir + files[i], bw=True)
    
    elevation_map = scharr(im)

    fig1 = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap='Greys_r');
    plt.subplot(1, 2, 2)
    plt.imshow(elevation_map, cmap='Greys_r');
    
## Gaussian filter

