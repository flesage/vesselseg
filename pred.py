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
from util import plot


if __name__=='__main__': 
    
    imagepath='/home/rdamseh/VascNet_supp/data/mouse10.tiff'
    segpath='/home/rdamseh/VascNet_supp/data/mouse10_seg.tiff'
    modelpath='results/VNet_BSC_31_01_2021_14_51'
    kernel_size=(64,64,64) #if all image at once: block_size=None
    
    # read image 
    im=skio.imread(imagepath)[0:96,0:96,0:96]
    im=mclahe.mclahe(im, kernel_size=kernel_size, adaptive_hist_range=True) # adaptive hist equalization
    s1, s2, s3 = np.array(im.shape).astype(float)
    
    # modeify image size on dim1, dim2 and dim3 to be multiples of k1, k2, k3 
    k1, k2, k3 = np.array(kernel_size).astype(float)
    d1=((k1*(s1%k1>0))-s1%k1)
    d2=((k2*(s2%k2>0))-s2%k2)
    d3=((k3*(s3%k3>0))-s3%k3)
    ss1 = d1+s1 
    ss2 = d2+s2 
    ss3 = d3+s3
    im=np.pad(im, ((0,int(d1)),(0,int(d2)),(0,int(d3))), constant_values=0) # add padding
    
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
    im = im[:-int(d1),:-int(d2),:-int(d3)]
    
    # plot with mayavi in 3D
    plot(im, out.numpy()>0.5)
    
    # plot 2d slices 
    seg=(skio.imread(segpath)>0).astype(int)
    from matplotlib import pyplot as plt
    plt.figure(); plt.imshow(im[50,:,:]); plt.title('image')
    plt.figure(); plt.imshow(seg[50,:,:]); plt.title('seg')
    plt.figure(); plt.imshow(out[50,:,:].numpy()); plt.title('pred')

                                                                      

