#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:58:54 2021

@author: rdamseh
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import time 
import h5py 
import numpy as np

from scipy.ndimage import gaussian_filter, median_filter, sobel
from scipy.signal.signaltools import wiener


def param_num(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) 
    print(params)

class ImageDataset(Dataset):
    '''Class that loads 3D images from a hdf5 file'''

    def __init__(self, path, length=None, augment=0):

        # Open hdf5 file
        self.data=h5py.File(path, 'r')
        self.image_size=np.shape(self.data['im0']) # get image size
        self.len_data = length
        print('-' * 89)
        print('Loading data...')

    def __len__(self):
        '''Return data length'''

        if self.len_data is None:
            self.len_data= len(self.data)
        return self.len_data

    def __getitem__(self, idx):
        '''
        return 2 batch (im & seg): each of shape (N, C, S1, S2, S3)
        # N is number of elements/images in batches (N='batch_size')
        # C is number of channels = 1
        # S1, S2, S3 is element/image size

        data.size() --> (S, N, E)
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Extract image and its corresponding segmentation ground truth
        im = np.array([self.data['im'+str(idx)]])
        # TODO: Possibly augment images and add noise to some of them to improve segmentation learning
        seg = np.array([self.data['seg'+str(idx)]])

        # Create pytorch tensor
        im = torch.FloatTensor(im)
        seg = torch.LongTensor(seg>0)
        im = im[:,None,:,:,:] # Add C axis
        seg = seg[:,None,:,:,:] # Add C axis

        return im, seg

def train(device,model, criterion, optimizer, scheduler, dataset, data_length, epoch, batch_size=1):
    '''
    dataset --> dataset class
    '''

    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time() # get start time
    loss_list=[]
    batch = 0 ##

    for data, targets in dataset:
        optimizer.zero_grad() # Reinitialize gradients for backpropagation

        data = data[:,0,:,:,:]
        targets = targets[:,0,:,:,:]
        # Send data to device
        data = data.to(device)
        targets = targets.to(device)
        # Apply segmentation model to data
        output = model(data)
        # Calculate segmentation loss
        loss = criterion(output, targets)
        loss.backward() # calculate gradients with backpropagation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) # prevent gradients from exceeding 0.5
        optimizer.step() # update model parameters according to gradients

        total_loss += loss.item() # add loss to batch total
        loss_list.append(loss.item()) # add loss to batch list

        # Print batch info 5 times per epoch
        log_interval = int(data_length / batch_size / 5)
        if log_interval == 0: ##
            log_interval = 1 ##
        if batch % log_interval == 0 and batch > 0:
            
            print('total_loss: ', total_loss)
            cur_loss = total_loss / log_interval # calculate current average loss
            elapsed = time.time() - start_time # calculate batch training time
            # print batch training info
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                    epoch, batch, data_length // batch_size, scheduler.get_last_lr()[0], ##dataset.len_train // batch_size, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time() # restart time
        
        batch += 1 ##
            
    return loss_list

def evaluate(device,model, criterion, dataset, data_length, batch_size=1):
    
    model.eval() # Turn on evaluation mode
    total_loss = 0.
    loss_list=[]
    
    with torch.no_grad(): # without backpropagation
        for data, targets in dataset:
            data = data[:,0,:,:,:]
            targets = targets[:,0,:,:,:]
            # Send data to device
            data = data.to(device)
            targets = targets.to(device)
            # Apply segmentation model to image
            output = model(data)
            # Calculate segmentation loss
            loss = criterion(output, targets)
            total_loss += loss.item() # Add loss to total
            loss_list.append(loss.item()) # Add loss to list
            average_loss = total_loss / (data_length/ batch_size) # Calculate average loss
    
    return average_loss, loss_list


class DiceLoss(nn.Module):
    '''Loss associated to Dice coefficient'''
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        inputs_numpy=inputs.detach().cpu().numpy() ##
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
        
        ## Total variation (added to loss for contours detection)
        ## to improve segmentation...
        sobelx = np.absolute(sobel(inputs_numpy,axis=0))
        sobely = np.absolute(sobel(inputs_numpy,axis=1))
        sobelz = np.absolute(sobel(inputs_numpy,axis=2))
        tv = np.sum(sobelx) + np.sum(sobely) + np.sum(sobelz)
        tv = tv * 3e-9
        ##

        return 1 - dice + tv

class IoULoss(nn.Module):
    '''Loss associated to Jaccard index'''
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU

class DiceBCELoss(nn.Module):
    '''Loss associated to Dice BCE coefficient'''
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth) 
 
        targets=targets.float()
        BCE = torch.functional.F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE

def save(model, epoch, optimizer, path):
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)

def model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def plot(a, b):
    '''
    a: image float
    b: label int
    '''
    def normalize(x):
        x=(x-x.min())/(x.max()-x.min())
        return x*255.0 
    
    try:  
        import VascGraph as vg
        vg.Tools.VisTools.visStack(np.array(b).astype(int), opacity=.2, color=(0,0,0))
        vg.Tools.VisTools.visVolume(normalize(np.array(a).astype(float)))
    except: print('--VascGraph package is not installed!')


def plotSeg(out):
    '''Plots 3D vascular image as a 3D graph'''
    
    try:  
        import VascGraph as vg
        vg.Tools.VisTools.visStack(np.array(out).astype(int), opacity=.2, color=(0,0,0))
    except: print('--VascGraph package is not installed!')

def preprocess(im):
    '''Preprocess TCO image to improve segmentation'''

    ImageDemasquee = []
    im=im*100 # Arbitrary scale factor to fit average training intensity
    im = gaussian_filter(im, sigma=3)
    for i in range(im.shape[0]): # for each plane
        # Subtraction of the median value for each 2D plane
        im[i,:,:] = im[i,:,:] - np.median(im[i,:,:].flatten())
        # Put low values at 0 for each plane
        im[i,:,:] = np.where(im[i,:,:] < 0.1 , 0 , im[i,:,:])
        # Reconstruct 3D image
        ImageDemasquee.append(im[i,:,:])
    im = np.array(ImageDemasquee)

    return im

if __name__=="__main__":
    pass