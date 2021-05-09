#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 21:09:53 2021

@author: rdamseh
"""
import skimage.io as skio
import mclahe
import torch
import numpy as np
from tqdm import tqdm
import os 

import argparse

def predict(imagepath, modelpath, kernel_size):

    # read image saved in TIF file
    try:
        im=skio.imread(imagepath)
    except:
        im=skio.imread(imagepath[:-1])
    #print('im')##
    #print(im.shape)##
    #print(im)##
    im=mclahe.mclahe(im, kernel_size=kernel_size, adaptive_hist_range=True) # adaptive hist equalization
    im=mclahe.mclahe(im, kernel_size=kernel_size, adaptive_hist_range=True) # adaptive hist equalization
    s1, s2, s3 = np.array(im.shape).astype(float)
    print('Input image size: '+str(im.shape))
    
    # modify image size on dim1, dim2 and dim3 to be multiples of k1, k2, k3 
    k1, k2, k3 = np.array(kernel_size).astype(float)
    d1=((k1*(s1%k1>0))-s1%k1)
    d2=((k2*(s2%k2>0))-s2%k2)
    d3=((k3*(s3%k3>0))-s3%k3)
    im=np.pad(im, ((0,int(d1)),(0,int(d2)),(0,int(d3))), constant_values=0) # add 0 padding
    print('Input image size after padding: '+str(im.shape))

    # check kernel size
    ks1=int(min(s1, k1))
    ks2=int(min(s2, k2))
    ks3=int(min(s2, k3))
    
    # extract image patches of kernel size
    patches=torch.FloatTensor(im.copy())
    patches=patches.unfold(0, ks1, ks1).unfold(1, ks2, ks2).unfold(2, ks3, ks3)
    unfold_shape=patches.shape
    print('unfold_shape:'+str(unfold_shape)) ##
    patches=patches.contiguous().view(-1, ks1, ks2, ks3).unsqueeze(1).unsqueeze(1)
    print('patches_shape:'+str(patches.shape)) ##
    
    # run prediction ...
    model=torch.load(modelpath) # get model parameters saved in a file
    model.eval() # turn on model evaluation mode
    with torch.no_grad(): # without backpropagation
        patches=[model(i) for i in tqdm(patches)] # apply model to each patch
        patches=torch.stack(patches)
    
    output_c = unfold_shape[0] * unfold_shape[3]
    output_h = unfold_shape[1] * unfold_shape[4]
    output_w = unfold_shape[2] * unfold_shape[5]
    out = patches.view(unfold_shape)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
    out = out.view(output_c, output_h, output_w)
    out = torch.sigmoid(out) # apply sigmoid function
    
    # remove padding
    out = out[:-int(d1),:-int(d2),:-int(d3)]
    #im = im[:-int(d1),:-int(d2),:-int(d3)]
    
    return out

# construct terminal parser
parser = argparse.ArgumentParser(description='VNet vascular sementation.')

parser.add_argument('-i', 
                    type=str,
                    help='image path: 3D angiogram sotred as .tif/.tiff file.')

parser.add_argument('-m', 
                    type=str,
                    help='model path: pytorch model parameters.')

parser.add_argument('-k',
                    nargs='+',
                    type=str,
                    help='kernel size. e.g. 64 64 64',
                    default='64 64 64')

parser.add_argument('-o', 
                    type=str,
                    help='output path.',
                    required=False,
                    default='')

parser.add_argument('-t', 
                    type=float,
                    help='prediction threshold, e.g. 0.5',
                    required=False,
                    default=0.0)

args = parser.parse_args()

# arguments
imagepath=args.i
modelpath=args.m
k=args.k
kernel_size=[int(i) for i in k]
threshold=args.t
outpath=args.o

# savepath
filename=imagepath.split('/')[-1]
if outpath !='':
    savepath=os.path.dirname(outpath+'/')+'/'+imagepath.split('/')[-1]
savepath=outpath+'LABEL_'+filename+'_test'

# predict segmentation
out=predict(imagepath, modelpath, kernel_size)

# apply threshold
if threshold>0 and threshold<1:
    out=(out.numpy()>threshold).astype('uint8')*255
else:
    out=(out.numpy()*255).astype('uint8')
#print('out') ##
#print(out.shape) #
#print(out) #
# save segmentation
print('Saving output: '+savepath)
skio.imsave(savepath, out) # save image in a TIF file
