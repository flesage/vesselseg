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
from scipy.ndimage import zoom
from tqdm import tqdm
from util import plot


if __name__=='__main__': 
    
    imagepath='/home/rdamseh/VascNet_supp/data/mouse10.tiff'
    modelpath='results/VNet_BSC_30_01_2021_20_48'
   
    kernel_size=(32,32,32) #if all image at once: block_size=None
    k1, k2, k3 = np.array(kernel_size).astype(float)
    
    # read image 
    im=skio.imread(imagepath)[0:96, 0:96, 0:96]
    #im=np.random.rand(32,100,100)
    im=mclahe.mclahe(im, kernel_size=kernel_size, adaptive_hist_range=True) # adaptive hist equalization
    s1, s2, s3 = np.array(im.shape).astype(float)
    
    # modeify image size on dim1, dim2 and dim3 to be multiples of k1, k2, k3 
    d1=((k1*(s1%k1>0))-s1%k1)
    d2=((k2*(s2%k2>0))-s2%k2)
    d3=((k3*(s3%k3>0))-s3%k3)
    ss1 = d1+s1 
    ss2 = d2+s2 
    ss3 = d3+s3
    #im0=zoom(im, (ss1/s1, ss2/s2, ss3/s3))
    im=np.pad(im, ((0,int(d1)),(0,int(d2)),(0,int(d3))), constant_values=0)
    
    # check kernal size
    ks1=int(min(s1, k1))
    ks2=int(min(s2, k2))
    ks3=int(min(s2, k3))
    
    # extract patches
    patches=torch.FloatTensor(im)
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
    seg = patches.view(unfold_shape)
    seg = seg.permute(0, 3, 1, 4, 2, 5).contiguous()
    seg = seg.view(output_c, output_h, output_w)
    seg = torch.sigmoid(seg)
    
    #patches=patches.view(unfold_shape)
    #im_py=patches.reshape(int(ss1), int(ss2), int(ss3)) 
    #im_py=fold((int(ss1), int(ss2), int(ss3)), (ks1, ks2, ks3))(patches)

    

    # im1=im_py0.numpy()
    # #im2=zoom(im1, (s1/ss1, s2/ss2, s3/ss3))
    # #model=torch.load(modelpath)
    plot(im, seg.numpy()>0.5)
    # from matplotlib import pyplot as plt
    # plt.figure(); plt.imshow(im[50,:,:])
    # plt.figure(); plt.imshow(seg[50,:,:].numpy())

                                                                      

