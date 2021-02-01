#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 10:24:54 2020

@author: rdamseh
"""
from util import dataset, train, evaluate, DiceLoss, DiceBCELoss, IoULoss, plot 
from magic_vnet import vnet
from datetime import datetime
import time
import os
import torch 
import numpy as np


if __name__=='__main__':

    if not os.path.isdir('results'):
        os.mkdir('results')
        
    datafile_train='/home/rdamseh/VascNet_supp/data/train_reg.h5' # path to training dataset
    datafile_test='/home/rdamseh/VascNet_supp/data/test_reg.h5' # path to testing dataset
    run_test=1 # if testing is needed

    epochs = 5 # The number of epochs
    batch_size= 1
    lr = 0.01 # learning rate
    criterion = IoULoss() # loss function: DiceLoss(), DiceBCELoss(), IoULoss()
    length=None # number of training/testing batches: keep None if want to train/test on whole dataset

    # ---------- init torch ---------------------------
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    #---------------------------------------------------

    # set of model that could be used:
        # "vnet.VNet" --> originl vnet
        # "vnet.VNet" "vnet.VNet_CSE" "vnet.VNet_BSC" "vnet.VNet_SCSE" "vnet.VNet_SSE" "vnet.VNet_MABN" "vnet.VNet_ASPP" 
        # "vnet.VBNet" "vnet.VBNet_CSE" "vnet.VBNet_BSC" "vnet.VBNet_SCSE" "vnet.VBNet_SSE"  "vnet.VBNet_ASPP" 
        # "vnet.NestVNet" "vnet.NestVNet_CSE" "vnet.NestVNet_BSC" "vnet.NestVNet_SCSE" "vnet.NestVNet_SSE" "vnet.VBNet_ASPP" 
        # "vnet.NestVBNet" "vnet.NestVBNet_CSE" "vnet.NestVBNet_SCSE" "vnet.NestVBNet_SSE"  "vnet.VBNet_ASPP" 
        # "vnet.SKVNet" "vnet.SKVNet_ASPP" "vnet.SK_NestVNet" "vnet.SK_NestVNet_ASPP" "vnet.NestVBNet_SSE"

    # ---------- defintion of the model to be used -----
    model=vnet.VNet_BSC(1,1, num_blocks=[3,3,3,3])
    model_name='VNet_BSC'
    model_name=model_name+'_'+datetime.today().strftime("%d_%m_%Y_%H_%M")
    #---------------------------------------------------
    
    #----------- read dataset --------------------------
    data=dataset(datafile_train, length=length, augment=0)
    #---------------------------------------------------

    #----------- training routine -----------------------
    optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)
    
    best_val_loss = float("inf")
    best_model = None
    loss_train=[]
    loss_val=[]
    best_epoch=0
    
    for epoch in range(1, epochs + 1):
        
        epoch_start_time = time.time()
        
        # train
        loss_tr = train(model=model, 
                        criterion=criterion, 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        dataset=data, 
                        epoch=epoch,
                        batch_size=batch_size)
        loss_train.append(loss_tr)
        
        # val
        val_loss, loss_v = evaluate(model=model, 
                                    criterion=criterion, 
                                    dataset=data,
                                    batch_size=batch_size)
        loss_val.append(loss_v)
            
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | val loss {:5.5f}'.format(epoch, (time.time() - epoch_start_time),
                                          val_loss))
        
        print(val_loss)
        print('-' * 89)
    
        # save if best validation loss
        if val_loss < best_val_loss:
            
            best_val_loss = val_loss
            best_model = model
            best_epoch=epoch
            torch.save(best_model, 'results/'+model_name)
            
        scheduler.step() 
    
    # save train/val loss curve
    np.savetxt('results/valLoss_'+model_name, np.array(loss_val))
    np.savetxt('results/trainLoss_'+model_name, np.array(loss_train))

    #----------- testing -----------------------
    if run_test:
        data_test=dataset(datafile_test, train_ratio=0, length=length)
        
        test_loss, loss_t = evaluate(model=model, 
                                    criterion=criterion, 
                                    dataset=data_test,
                                    batch_size=batch_size)
        print('-' * 89)
        print('Test loss: {:5.5f}'.format(test_loss))
        np.savetxt('results/testLoss_'+model_name, np.array(loss_t))

    #----------- test example ---------------------------
    # test_model = model
    # datafile='/home/rdamseh/VascNet_supp/data/test_reg.h5'
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

