#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:24:54 2020

@author: rdamseh
"""
from util import ImageDataset, train, evaluate, DiceLoss, DiceBCELoss, IoULoss, plot 
from magic_vnet import vnet
from datetime import datetime
import time
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchio as tio
import numpy as np
import h5py 


if __name__=='__main__':

    if not os.path.isdir('results'):
        os.mkdir('results')
        
    script_dir = os.path.dirname(__file__)
    datafile_train=os.path.join(script_dir, ("VascNet_supp\\2PM_data\\train_reg.h5")) #'/home/rdamseh/VascNet_supp/data/train_reg.h5' # path to training dataset
    datafile_test=os.path.join(script_dir, ("VascNet_supp\\2PM_data\\test_reg.h5")) #'/home/rdamseh/VascNet_supp/data/test_reg.h5' # path to testing dataset
    run_test=0 # 1 if testing is needed

    epochs = 5 # The number of epochs
    batch_size= 1
    lr = 0.001 # learning rate
    criterion = IoULoss() # loss function: DiceLoss(), DiceBCELoss(), IoULoss()
    length=10 ##None # number of training/testing volumes: keep None if want to train/test on whole dataset

    # ---------- init torch ---------------------------
    torch.manual_seed(0) # generate random seed
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #---------------------------------------------------

    # set of models that can be used:
        # "vnet.VNet" --> originl vnet
        # "vnet.VNet" "vnet.VNet_CSE" "vnet.VNet_BSC" "vnet.VNet_SCSE" "vnet.VNet_SSE" "vnet.VNet_MABN" "vnet.VNet_ASPP" 
        # "vnet.VBNet" "vnet.VBNet_CSE" "vnet.VBNet_BSC" "vnet.VBNet_SCSE" "vnet.VBNet_SSE"  "vnet.VBNet_ASPP" 
        # "vnet.NestVNet" "vnet.NestVNet_CSE" "vnet.NestVNet_BSC" "vnet.NestVNet_SCSE" "vnet.NestVNet_SSE" "vnet.VBNet_ASPP" 
        # "vnet.NestVBNet" "vnet.NestVBNet_CSE" "vnet.NestVBNet_SCSE" "vnet.NestVBNet_SSE"  "vnet.VBNet_ASPP" 
        # "vnet.SKVNet" "vnet.SKVNet_ASPP" "vnet.SK_NestVNet" "vnet.SK_NestVNet_ASPP" "vnet.NestVBNet_SSE"

    # ---------- defintion of the model to be used -----
    model=vnet.VNet_BSC(1,1, num_blocks=[3,3,3,3]) ###VNet_BSC
    model = model.to(device) ##
    model_name='VNet_BSC' ###
    model_name=model_name+'_'+str(length)+'images_'+str(batch_size)+'batchsize_'+datetime.today().strftime("%d_%m_%Y_%H_%M")
    #---------------------------------------------------
    
    #----------- read dataset --------------------------
    dataset=ImageDataset(datafile_train, length=length, augment=0) ##
    #---------------------------------------------------

    #----------- training routine -----------------------
    # define optimizer to adjust model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) ##RMSprop
    # define scheduler to adjust learning rate according to epoch
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
    
    best_val_loss = float("inf")
    best_model = None
    loss_train=[]
    loss_val=[]
    best_epoch=0
    total_time=0
    
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time() # get start time
        
        # train model
        train_ratio=.7 ##
        idx1=int(len(dataset)*train_ratio) ##
        train_dataset, val_dataset = random_split(dataset, [idx1,(len(dataset)-idx1)])  ##

        print('Images for training: '+str(len(train_dataset))+' | Images for Validation: '+str(len(val_dataset)))
        print('-' * 89)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
        ##
        loss_tr = train(device=device,model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, ##
                        dataset=train_loader, data_length=len(train_dataset),epoch=epoch, batch_size=batch_size) ##
        loss_train.append(loss_tr) # add training loss to epoch total
        
        # validate model
        val_loss, loss_v = evaluate(device=device,model=model,criterion=criterion, dataset=val_loader, data_length=len(val_dataset), batch_size=batch_size) ##
        loss_val.append(loss_v) # add validation loss to epoch total
        
        epoch_elapsed = time.time() - epoch_start_time # calculate epoch training time
        total_time += epoch_elapsed
        # print epoch training info
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.5f}'.format(epoch, epoch_elapsed, val_loss))
        print(val_loss)
        print('-' * 89)
    
        # save model if best validation loss
        if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            best_model = model
            best_epoch=epoch
            torch.save(best_model, 'results/'+model_name) # save model in file
            
        scheduler.step() # adjust learning rate
    
    # print total time
    print('-' * 89)
    print('Total training time: {:5.2f}s'.format(total_time))
    
    # save train/val loss curve
    np.savetxt('results/valLoss_'+model_name, np.array(loss_val))
    np.savetxt('results/trainLoss_'+model_name, np.array(loss_train))

    #----------- testing -----------------------
    if run_test:
        if length > 255: ##
            length = 255 ## max number of images in test file
        
        data_test=ImageDataset(datafile_test, length=length, augment=0) ##
        test_loader = DataLoader(dataset=data_test, batch_size=batch_size, shuffle=False) ##
        
        test_loss, loss_t = evaluate(device=device,model=model,criterion=criterion, dataset=test_loader, data_length=len(data_test), batch_size=batch_size) ##
        print('-' * 89)
        print('Test loss: {:5.5f}'.format(test_loss))
        np.savetxt('results/testLoss_'+model_name, np.array(loss_t))

    #----------- test example ---------------------------
    # test_model = model
    # datafile='C:\\Users\\Mathieu\\Documents\\VascNet_supp\\2PM_data\\test_reg.h5'  #'/home/rdamseh/VascNet_supp/data/test_reg.h5'
    # data_test=dataset(datafile_test)
    # idx=100
    # im, seg = data_test.get_batch(idx, 1) # return test batch with index 'idx'
    # with torch.no_grad():
    #     out=torch.sigmoid(test_model(im))
    #     out=out.cpu().numpy()
    #     out=out[0,0,:,:,:]
    # im_=im.cpu().numpy()[0,0,:,:,:]
    # seg_=seg.cpu().numpy()[0,0,:,:,:]
    # plot(im_, seg_>0)
    # plot(im_, out>.9)
        
    # from matplotlib import pyplot as plt
    # fig, axes = plt.subplots(1,3)
    # for ax, i, l in zip(axes, [im_, seg_, out], ['im','seg','pred']):
    #     ax.imshow(i[32,:,:], label=l)