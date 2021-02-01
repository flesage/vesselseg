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

    # read image 
    try:
        im=skio.imread(imagepath)
    except:
        im=skio.imread(imagepath[:-1])

    im=mclahe.mclahe(im, kernel_size=kernel_size, adaptive_hist_range=True) # adaptive hist equalization
    im=mclahe.mclahe(im, kernel_size=kernel_size, adaptive_hist_range=True) # adaptive hist equalization
    s1, s2, s3 = np.array(im.shape).astype(float)
    print('Input image size: '+str(im.shape))
    
    # modeify image size on dim1, dim2 and dim3 to be multiples of k1, k2, k3 
    k1, k2, k3 = np.array(kernel_size).astype(float)
    d1=((k1*(s1%k1>0))-s1%k1)
    d2=((k2*(s2%k2>0))-s2%k2)
    d3=((k3*(s3%k3>0))-s3%k3)
    im=np.pad(im, ((0,int(d1)),(0,int(d2)),(0,int(d3))), constant_values=0) # add padding
    print('Input image size after padding: '+str(im.shape))

    # check kernal size
    ks1=int(min(s1, k1))
    ks2=int(min(s2, k2))
    ks3=int(min(s2, k3))
    
    # extract patches
    patches=torch.FloatTensor(im.copy())
    patches=patches.unfold(0, ks1, ks1).unfold(1, ks2, ks2).unfold(2, ks3, ks3)
    unfold_shape=patches.shape
    patches=patches.contiguous().view(-1, ks1, ks2, ks3).unsqueeze(1).unsqueeze(1)
    
    # run prediction ...
    model=torch.load(modelpath)
    model.eval()
    with torch.no_grad():
        patches=[model(i) for i in tqdm(patches)]
        patches=torch.stack(patches)
    
    output_c = unfold_shape[0] * unfold_shape[3]
    output_h = unfold_shape[1] * unfold_shape[4]
    output_w = unfold_shape[2] * unfold_shape[5]
    out = patches.view(unfold_shape)
    out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
    out = out.view(output_c, output_h, output_w)
    out = torch.sigmoid(out)
    
    # remove padding
    out = out[:-int(d1),:-int(d2),:-int(d3)]
    #im = im[:-int(d1),:-int(d2),:-int(d3)]
    
    return out


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
                    help='kernal size. e.g. 64 64 64',
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

# argmanets
imagepath=args.i
modelpath=args.m
k=args.k;
kernel_size=[int(i) for i in k]
threshold=args.t
outpath=args.o

# savepath
filename=imagepath.split('/')[-1]
if outpath !='':
    savepath=os.path.dirname(outpath+'/')+'/'+imagepath.split('/')[-1]
savepath=outpath+'LABEL_'+filename

#predict
out=predict(imagepath, modelpath, kernel_size)

#save
if threshold>0 and threshold<1:
    out=(out.numpy()>threshold).astype('uint8')*255
else:
    out=(out.numpy()*255).astype('uint8')
print('Saving output: '+savepath)
skio.imsave(savepath, out)


