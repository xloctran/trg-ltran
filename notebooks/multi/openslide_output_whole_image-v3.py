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

# # Code

# +
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import scipy.signal
import argparse


import sklearn.feature_extraction.image

import matplotlib.cm

import torch

from torchvision.models import DenseNet

from tqdm.autonotebook import tqdm

from  skimage.color import rgb2gray

os.environ['PATH'] = 'C:\\research\\openslide\\bin' + ';' + os.environ['PATH'] #can either specify openslide bin path in PATH, or add it dynamically
import openslide


# +
parser = argparse.ArgumentParser(description='Make output for entire image using Unet')
parser.add_argument('input_pattern',
                    help="input filename pattern. try: *.png, or tsv file containing list of files to analyze",
                    nargs="*")

parser.add_argument('-p', '--patchsize', help="patchsize, default 256", default=256, type=int)
parser.add_argument('-s', '--batchsize', help="batchsize for controlling GPU memory usage, default 10", default=10, type=int)
parser.add_argument('-o', '--outdir', help="outputdir, default ./output/", default="./output/", type=str)
parser.add_argument('-r', '--resize', help="resize factor 1=1x, 2=2x, .5 = .5x", default=1, type=float)
parser.add_argument('-m', '--model', help="model", default="best_model.pth", type=str)
parser.add_argument('-i', '--gpuid', help="id of gpu to use", default=0, type=int)
parser.add_argument('-f', '--force', help="force regeneration of output even if it exists", default=False,
                    action="store_true")
parser.add_argument('-b', '--basepath',
                    help="base path to add to file names, helps when producing data using tsv file as input",
                    default="", type=str)

args = parser.parse_args(["-mbrca1_densenet_best_model.pth"])
# -


device = torch.device(args.gpuid if args.gpuid!=-2 and torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666
model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                 num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                 drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)
model.load_state_dict(checkpoint["model_dict"])
model.eval()
print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

fname=r"D:\research\brca1\mib1\BRCA03_MIB1.mrxs"
osh  = openslide.OpenSlide(fname)


osh.level_dimensions


def divide_batch(l, n): 
    for i in range(0, l.shape[0], n):  
        yield l[i:i + n,::] 


#add mask creation which skips parts of image
mask_level = 8
img = osh.read_region((0, 0), mask_level, osh.level_dimensions[mask_level])
img = np.asarray(img)[:, :, 0:3]
imgg=rgb2gray(img)
mask=np.bitwise_and(imgg>0 ,imgg <200/255)

plt.imshow(mask)



#blue, orange, green, red
#"Stroma","Tumor","Immune cells","Other"]
cmap= matplotlib.cm.tab10

# %matplotlib inline

# +
level=0
ds=int(osh.level_downsamples[level])

patch_size=256
stride_size=patch_size//4
tile_size=stride_size*8*2
tile_pad=patch_size-stride_size
nclasses=3
batch_size=64 #should be a power of 2


shape=osh.level_dimensions[level]
shaperound=[((d//tile_size)+1)*tile_size for d in shape]

#npmm = np.memmap('npmm9.dat',mode='w+', dtype=np.uint8,shape=(osh.level_dimensions[dim][1]//patch_size,osh.level_dimensions[dim][0]//patch_size,3))
#npmm=np.zeros((osh.level_dimensions[dim][1]//patch_size,osh.level_dimensions[dim][0]//patch_size,3))
npmm=np.zeros((shaperound[1]//stride_size,shaperound[0]//stride_size,3),dtype=np.uint8)
for y in tqdm(range(0,osh.level_dimensions[0][1],round(tile_size * osh.level_downsamples[level])), desc="outer"):
    for x in tqdm(range(0,osh.level_dimensions[0][0],round(tile_size * osh.level_downsamples[level])), desc=f"innter {y}", leave=False):

        #if skip
        
        #maskx=int(x//(osh.level_dimensions[level][0]/osh.level_dimensions[mask_level][0]))
        #masky=int(y//(osh.level_dimensions[level][1]/osh.level_dimensions[mask_level][1]))
        
        maskx=int(x//osh.level_downsamples[mask_level])
        masky=int(y//osh.level_downsamples[mask_level])
        
        
        if(maskx>= mask.shape[1] or masky>= mask.shape[0] or not mask[masky,maskx]): #need to handle rounding error 
            continue
        
        
        output = np.zeros((0,nclasses,patch_size//patch_size,patch_size//patch_size))
        io = np.asarray(osh.read_region((x, y), level, (tile_size+tile_pad,tile_size+tile_pad)))[:,:,0:3] #trim alpha
        
        arr_out=sklearn.feature_extraction.image.extract_patches(io,(patch_size,patch_size,3),stride_size)
        arr_out_shape = arr_out.shape
        arr_out = arr_out.reshape(-1,patch_size,patch_size,3)
        
        for batch_arr in divide_batch(arr_out,batch_size):
        
            arr_out_gpu = torch.from_numpy(batch_arr.transpose(0, 3, 1, 2) / 255).type('torch.FloatTensor').to(device)

            # ---- get results
            output_batch = model(arr_out_gpu)

             # --- pull from GPU and append to rest of output 
            output_batch = output_batch.detach().cpu().numpy()
#            a = batch_arr.reshape(batch_arr.shape[0],-1,arr_out.shape[-1])
#            output_batch=a.mean(axis=1)[:,:,None,None]
            output_batch_color=cmap(output_batch.argmax(axis=1), alpha=None)[:,0:3]
            output = np.append(output,output_batch_color[:,:,None,None],axis=0)
        
        output = output.transpose((0, 2, 3, 1))
        #turn from a single list into a matrix of tiles
        output = output.reshape(arr_out_shape[0],arr_out_shape[1],patch_size//patch_size,patch_size//patch_size,output.shape[3])
        
        #turn all the tiles into an image
        output=np.concatenate(np.concatenate(output,1),1)
        #print(y//stride_size//ds,y//stride_size//ds+tile_size//stride_size,x//stride_size//ds,x//stride_size//ds+tile_size//stride_size)
        npmm[y//stride_size//ds:y//stride_size//ds+tile_size//stride_size,x//stride_size//ds:x//stride_size//ds+tile_size//stride_size,:]=output*255 #need to save uint8
# -

from skimage.external.tifffile import TiffWriter
with TiffWriter('redux_s4xx.tif', bigtiff=True, imagej=True) as tif:
    tif.save(npmm, compress=6, tile=(256,256) )

npmm.shape

# %load_ext line_profiler
# %lprun -f doit doit()
