#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 11:43:43 2020

@author: vient
"""

import numpy as np
from tqdm import tqdm
import time
from sklearn.decomposition import PCA
from random import randrange

def RMSE(pred,trueobs):
    return np.sqrt(np.nanmean((pred-trueobs)**2))

def DinEOF_appli(x,n_EOF):
    '''Application of DinEOF with n_EOF compnents to a dataset,
    returning the DinEOF reconstructed dataset.
    x_coord = Coordonates of sea points (non _ nan values)
    '''
    pca            = PCA(n_EOF)
    x_coord        = np.where(~np.isnan(x[0]))
    x_truepts      = x[:,x_coord[0],x_coord[1]].reshape(x.shape[0],-1)
    x_reconstruct  = pca.inverse_transform(pca.fit_transform(x_truepts))

    x[:,x_coord[0],x_coord[1]] = x_reconstruct

    return x 

def DinEOF_iter(x_Train,x_Gap,x_Pred,OI,n_EOF,n_Iter):
    x_fill = np.copy(x_Pred)
    history = dict()
    for i in tqdm(range(n_Iter)):
        if i == 0:
            x_fill[np.where(np.isnan(x_Gap))] = OI[np.where(np.isnan(x_Gap))]
        else: 
            x_fill[np.where(np.isnan(x_Gap))] = x_interp[np.where(np.isnan(x_Gap))]
        x_interp = DinEOF_appli(np.concatenate((x_Train,x_fill)),n_EOF)[len(x_train):]
        history[i]=RMSE(x_Pred,x_interp)
    return x_interp,history

dataset  = np.load("/home/vient/Thèse/Data/Data_ZOI/Dataset_64_ZOI.npy",allow_pickle='TRUE').item()
dataH    = dataset['CSED_Hourly']
mask     = dataset['Cloud_Daily']
lat_grid = dataset['Lat_ZOI']
lon_grid = dataset['Lon_ZOI']
OI       = np.load('/home/vient/Thèse/Python/Script_Python/Optimal_Interpolation/pred_OI.npy')

Save   = True
flagML = False

def prepdata(xH,mask,N_Catalog=0):
    if N_Catalog!=0:
        test=np.zeros(mask[-365:].shape)
        for i in range(0,len(test)):
            test[i]=xH[26304+i*24+12].reshape(test.shape[1],test.shape[2])

        train=np.zeros((N_Catalog*365,xH.shape[1],xH.shape[2]))

        for i in range(N_Catalog):
            for j in range(len(test)):
                h=randrange(0,24)
                train[i*365+j]=xH[i*365+j*24+h]
    else:
        xD=np.empty((xH.shape[0]//24,xH.shape[1],xH.shape[2]))
        for i in range(len(xD)):
            xD[i]=xH[12+i*24]
    mask_train = mask[:-100]
    mask_pred  = mask[-100:]
    x_train    = xD[:len(mask_train)]
    y_pred     = xD[len(mask_train):len(mask_train)+len(mask_pred)]
    return mask_train,mask_pred,x_train,y_pred

mask_train,mask_pred,x_train,y_pred = prepdata(dataH,mask)
print('Starting DinEOF interpolation')
pred,hist =DinEOF_iter(x_train,mask_pred,y_pred,OI,0.99,n_Iter = 12)
print('DinEOF interpolation ended with RMSE : '+str(np.sqrt(np.nanmean((pred - y_pred)**2))))

if Save:
    np.save('pred_DinEOF',pred)
    print('Prediction saved')