#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 19:58:54 2021

@author: rdamseh
"""

import torch
import torch.nn as nn
import time 
import h5py 
import numpy as np



def param_num(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters]) 
    print(params)
    
class dataset:
    
    def __init__(self, path, train_ratio=.7, length=None, augment=0):
       
        self.data=h5py.File(path, 'r')
        if length is not None:
            self.len_data=length
        else:
            self.len_data= len(self.data)
        self.image_size=np.shape(self.data['im0'])

        self.all_ind=np.arange(self.len_data)
        
        idx1=int(self.len_data*train_ratio)
        self.train_ind=self.all_ind[0:idx1]
        self.val_ind=self.all_ind[idx1:]
        
        self.len_train=len(self.train_ind)
        self.len_val=len(self.val_ind)
        
        
    def get_batch(self, i, batch_size, which='train'):
        
        '''
        return 2 batch (im & seg): each of shape (N, C, S1, S2, S3)
        # N is number of elemts/images in batches (N='batch_size')
        # C is number of channels = 1
        # S1, S2, S3 is element/image size

        data.size() --> (S, N, E)
        '''
        
        len_min = max(self.len_train, self.len_val) 
        len_batch = min(batch_size, len_min - 1 - i)
        ind=np.arange(i, i+len_batch)
        
        if which=='train':
            ind=self.train_ind[ind]
        if which=='val':
            ind=self.val_ind[ind]

        im=np.array([self.data['im'+str(i)] for i in ind])
        seg=np.array([self.data['seg'+str(i)] for i in ind])

        im=torch.FloatTensor(im)
        seg=torch.LongTensor(seg>0)

        return im[:,None,:,:,:], seg[:,None,:,:,:]


def train(model, criterion, optimizer, scheduler, dataset, epoch, batch_size=1):
    
    '''
    dataset --> dataset class
    '''
    
    model.train() # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    loss_list=[]
    
    for batch, i in enumerate(range(0, dataset.len_train - 1, batch_size)):
    
        data, targets = dataset.get_batch(i, batch_size, which='train')
        
        optimizer.zero_grad()
        output = model(data)  
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        loss_list.append(loss.item())
        log_interval = int(dataset.len_train / batch_size / 5)
        
        if batch % log_interval == 0 and batch > 0:
            
            print('total_loss: ', total_loss)
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f}'.format(
                    epoch, batch, dataset.len_train // batch_size, scheduler.get_last_lr()[0],
                    elapsed * 1000 / log_interval, cur_loss))
            total_loss = 0
            start_time = time.time()
            
    return loss_list

def evaluate(model, criterion, dataset, batch_size=1):
    
    model.eval() # Turn on the evaluation mode
    total_loss = 0.
    loss_list=[]
    
    with torch.no_grad():
        
        for i in range(0, dataset.len_val - 1, batch_size):
            
            data, targets = dataset.get_batch(i, batch_size, which='val')            
            output = model(data)            
            loss = criterion(output, targets).cpu().item()
            total_loss += loss
            loss_list.append(loss)
                        
    return total_loss / (dataset.len_val/ batch_size), loss_list



class DiceLoss(nn.Module):
    
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class IoULoss(nn.Module):
    
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
    
if __name__=="__main__":
    pass