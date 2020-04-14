# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Making Output for the Multiresolution Network
#
# The network definition is stored in the module **MultiResSmallNetwork** in the directory src/models/
#
# In this Notebook, we want to generate the output for a part of an WSI at 5x. We start by specifying some global variables. 

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


from tqdm.autonotebook import tqdm

from  skimage.color import rgb2gray
import PIL

import glob

import dill as pickle

from skimage.color import rgb2gray, rgb2hed

from skimage.measure import * 
from skimage.filters import *
from skimage.morphology import *
import time


# -

def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 


# create a function to make output from the image, mask, set of points
def MakeOutput(model, device, img, centers, patch_size, patch_size_res2, batch_size):
    npatches=len(centers)
    arr_out_res1 = np.zeros((npatches,3,patch_size,patch_size))
    arr_out_res2 = np.zeros((npatches,3,patch_size,patch_size))
    img_transform = Compose([
       ToTensor()
    ])
    rs=[]
    cs=[]
    for i, (r, c) in tqdm(enumerate(centers), total = len(centers)):
        r=int(round(r))
        c=int(round(c))
        rs.append(r)
        cs.append(c)

        imgres1 = img[r-patch_size//2:r+patch_size//2,c-patch_size//2:c+patch_size//2,:]
            
        imgres2 = img[r-patch_size_res2//2:r+patch_size_res2//2,c-patch_size_res2//2:c+patch_size_res2//2,:]
        imgres2 = cv2.resize(imgres2,(patch_size,patch_size), interpolation=PIL.Image.BICUBIC) #resize it as specified above

    
        arr_out_res1[i,:,:,:] = img_transform(image=imgres1)["image"]
        arr_out_res2[i,:,:,:] = img_transform(image=imgres2)["image"]
    clusterids = []
    for batch_arr_res1, batch_arr_res2 in tqdm(zip(divide_batch(arr_out_res1,batch_size),divide_batch(arr_out_res2,batch_size))):

        #arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)
        arr_out_gpu_res1 =  torch.from_numpy(batch_arr_res1).type('torch.FloatTensor').to(device)
        arr_out_gpu_res2 =  torch.from_numpy(batch_arr_res2).type('torch.FloatTensor').to(device)

        # ---- get results
        clusterids.append(torch.argmax( model.dualfoward(arr_out_gpu_res1,arr_out_gpu_res2),dim=1).detach().cpu().numpy())
    clusterids=np.hstack(clusterids)
    return clusterids


def OutputMasks(mask, regions, centers, index, clusterids):
    result = np.zeros(mask.shape, dtype=int)
    for i in range(len(index)):
        for coord in list(regions[index[i]].coords):
            r, c = coord
            result[r, c] = clusterids[i] + 1
    return result


def Preprocess(img, resize, mirror_pad_size, patch_size_res2):
    img= cv2.resize(img,(0,0),fx=resize,fy=resize, interpolation=PIL.Image.BICUBIC) #resize it as specified above
    img = np.pad(img, [(mirror_pad_size, mirror_pad_size), (mirror_pad_size, mirror_pad_size), (0, 0)], mode="reflect")
    #create the coresponding mask by using hematoxylin
    #hed=rgb2hed(img)
    mask=img[:, :, 2] < 241
    # remove the region near the edge
    mask[0:patch_size_res2,:]=0
    mask[:,0:patch_size_res2]=0
    mask[:,-patch_size_res2-1:]=0
    mask[-patch_size_res2-1:,:]=0
    mask=remove_small_objects(mask,150)

    mask[img.sum(axis=2)<100]=0

    mask[img.sum(axis=2)>700]=0

    
    return img, mask 


def CentersSLIC(regions, mask): 
    centers = []
    index = []
    for i, region in enumerate(regions):
        (r, c) = region.centroid
        r, c = int(round(r)), int(round(c))
        if mask[r, c]!=0: 
            index.append(i)
            centers.append((r, c))
    return index, centers


def Intersection(lst1, lst2):  
    return list(set(lst1) & set(lst2))


def saveList(myList,filename):
    # the filename should mention the extension 'npy'
    np.save(filename,myList)
    print("Saved successfully!")


def loadList(filename):
    # the filename should mention the extension 'npy'
    tempNumpyArray=np.load(filename)
    return tempNumpyArray.tolist()




