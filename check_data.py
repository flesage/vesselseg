#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 22:00:15 2021

@author: rdamseh
"""
import h5py
import SimpleITK as sitk
from tqdm import tqdm
import numpy as np
import scipy.ndimage as nd

def normalize(x):
    x=(x-x.min())/(x.max()-x.min())
    return x*255.0


if __name__=='__main__':
    
    
    data_path='data/train.h5'
    f=h5py.File(data_path, 'r')
    ind=list(f.keys())
    ims=[i for i in ind if i[0:3]!='seg']
    sgs=[i for i in ind if i[0:3]=='seg']
    
    
    a=np.array(f['im100'])
    a=normalize(a)
    b=np.array(f['seg100'])
    b=normalize(b)
    
    try:  
        import VascGraph as vg
        vg.Tools.VisTools.visStack(np.array(b).astype(int), opacity=.2, color=(0,0,0))
        vg.Tools.VisTools.visVolume(normalize(np.array(a).astype(float)))
    except:
        print('--VascGraph package is not installed!')
        
        
    