#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 16:29:49 2020

@author: rdamseh
"""
import magic_vnet as vnet
import numpy as np
import torch

def model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

# set of model that could be used:
    # "vnet.VNet" --> originl vnet
    # "vnet.VNet" "vnet.VNet_CSE" "vnet.VNet_BSC" "vnet.VNet_SCSE" "vnet.VNet_SSE" "vnet.VNet_MABN" "vnet.VNet_ASPP" 
    # "vnet.VBNet" "vnet.VBNet_CSE" "vnet.VBNet_BSC" "vnet.VBNet_SCSE" "vnet.VBNet_SSE"  "vnet.VBNet_ASPP" 
    # "vnet.NestVNet" "vnet.NestVNet_CSE" "vnet.NestVNet_BSC" "vnet.NestVNet_SCSE" "vnet.NestVNet_SSE" "vnet.VBNet_ASPP" 
    # "vnet.NestVBNet" "vnet.NestVBNet_CSE" "vnet.NestVBNet_SCSE" "vnet.NestVBNet_SSE"  "vnet.VBNet_ASPP" 
    # "vnet.SKVNet" "vnet.SKVNet_ASPP" "vnet.SK_NestVNet" "vnet.SK_NestVNet_ASPP" "vnet.NestVBNet_SSE"

model=vnet.SKVNet(1,1, num_blocks=[3,3,3,3])
print('model size:', model_size(model))

try:
    from torchviz import make_dot
    im=torch.rand(1,1,64,64,64)
    dot=make_dot(model(im))
    dot.render("model")
except: 
    print('Torchviz is not istalled')
