# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + tags=["parameters"]
dataname="trg_multi"

#files=glob.glob('*.png')
#fname=files[3]
fname="../slide-2019-09-24T09-52-40-R1-S1.mrxs.png"
# -

class_names=["Fat", "Muscular", "Vessle", "Gland", "Stroma", "Tumor", "Necrosis", "Epithelium"]
nclasses=len(class_names)

# +
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import argparse
from torch import nn
from torchsummary import summary

from skimage import color 

from albumentations import *
from albumentations.pytorch import ToTensor

import sklearn.feature_extraction.image

import matplotlib.cm

import torch

from torchvision.models import DenseNet

from tqdm.autonotebook import tqdm

from  skimage.color import rgb2gray
import PIL

import glob

import dill as pickle

from skimage.color import rgb2gray, rgb2hed

from skimage.measure import * 
from skimage.filters import *
from skimage.morphology import *

# -


def blend2Images(img, mask):
    if (img.ndim == 3):
        img = color.rgb2gray(img)
    if (mask.ndim == 3):
        mask = color.rgb2gray(mask)
    img = img[:, :, None] * 1.0  # can't use boolean
    mask = mask[:, :, None] * 1.0
    out = np.concatenate((mask, img, mask), 2)
    return out


# +
resize_resolutions=[1,.25] #resize input image, base and second

patch_size=64
patch_size_res2 = int(patch_size  * 1/resize_resolutions[1])
mirror_pad_size = patch_size_res2

batch_size=128 #should be a power of 2

cmap= matplotlib.cm.tab10

# -

device = torch.device('cuda')
checkpoint = torch.load("./multi/trg_multi_simple_multires_best_model.pth", map_location=lambda storage, loc: storage)

from  MultiResSmallNetwork import MergeSmallNetworks
model=MergeSmallNetworks(nclasses = nclasses, outputsize = 8 ).to(device)
model.load_state_dict(checkpoint["model_dict"])

summary(model.res1,input_size=(3,64,64))

summary(model.res1.encoder,input_size=(3,64,64))


def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 


img = cv2.cvtColor(cv2.imread(fname),cv2.COLOR_BGR2RGB)
img= cv2.resize(img,(0,0),fx=resize_resolutions[0],fy=resize_resolutions[0], interpolation=PIL.Image.BICUBIC) #resize it as specified above
img = np.pad(img, [(mirror_pad_size, mirror_pad_size), (mirror_pad_size, mirror_pad_size), (0, 0)], mode="reflect")

hed=rgb2hed(img)
mask=hed[:,:,0]>-.8
mask=remove_small_objects(mask,150)

# +
mask[img.sum(axis=2)<100]=0

mask[img.sum(axis=2)>700]=0

mask[0:patch_size_res2,:]=0;
mask[:,0:patch_size_res2]=0;

mask[:,-patch_size_res2-1:]=0
mask[-patch_size_res2-1:,:]=0
# -

plt.imshow(mask)


def random_subset(a, b, nitems):
    assert len(a) == len(b)
    idx = np.random.randint(0,len(a),nitems)
    return a[idx], b[idx]


# +
max_number_samples=50000
[rs,cs]=mask.nonzero()
[rs,cs]=random_subset(rs,cs,max_number_samples)

centers = [ (r,c) for r,c in zip(rs,cs)]
npatches=len(centers)

arr_out_res1 = np.zeros((npatches,3,patch_size,patch_size))
arr_out_res2 = np.zeros((npatches,3,patch_size,patch_size))
# -

img_transform = Compose([
       ToTensor()
    ])

# +
rs=[]
cs=[]
for i, (r,c) in tqdm(enumerate(centers), total = len(centers)):
    r=int(round(r))
    c=int(round(c))
    rs.append(r)
    cs.append(c)

    imgres1 = img[r-patch_size//2:r+patch_size//2,c-patch_size//2:c+patch_size//2,:]
            
    imgres2 = img[r-patch_size_res2//2:r+patch_size_res2//2,c-patch_size_res2//2:c+patch_size_res2//2,:]
    imgres2 = cv2.resize(imgres2,(patch_size,patch_size), interpolation=PIL.Image.BICUBIC) #resize it as specified above

    
    arr_out_res1[i,:,:,:] = img_transform(image=imgres1)["image"]
    arr_out_res2[i,:,:,:] = img_transform(image=imgres2)["image"]

    
    
# -

clusterids=[]
for batch_arr_res1, batch_arr_res2 in tqdm(zip(divide_batch(arr_out_res1,batch_size),divide_batch(arr_out_res2,batch_size))):

    #arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
    arr_out_gpu_res1 =  torch.from_numpy(batch_arr_res1).type('torch.FloatTensor').to(device)
    arr_out_gpu_res2 =  torch.from_numpy(batch_arr_res2).type('torch.FloatTensor').to(device)

    # ---- get results
    clusterids.append(torch.argmax( model.dualfoward(arr_out_gpu_res1,arr_out_gpu_res2),dim=1).detach().cpu().numpy())
clusterids=np.hstack(clusterids)

# %matplotlib notebook
plt.rcParams["figure.figsize"] = [10, 20]
fig, ax = plt.subplots()
ax.imshow(img)
ax.scatter(cs,rs,c=clusterids,cmap=cmap,s=1)

from collections import Counter
Counter(clusterids)


