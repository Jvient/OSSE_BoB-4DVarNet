#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 14:28:32 2020

@author: vient
"""
import numpy as np
np.seterr(divide = 'ignore')
#numpy.seterr(divide = 'warn')
# esle befor loglik : with np.errstate(divide='ignore'):
from sklearn.decomposition import PCA
from random import randrange

import os

os.chdir('/home/vient/Thèse/Python/Script_Python/MAJ_AnDA0720/AnDA-master')
# analog data assimilation
from AnDA_codes.AnDA_analog_forecasting import AnDA_analog_forecasting
from AnDA_codes.AnDA_data_assimilation import AnDA_data_assimilation
from AnDA_codes.AnDA_stat_functions import AnDA_RMSE

### PARAMETERS ###

flagAnalogs    = 'global' # resolution scale ['global','local']
flagMethod     = 'AnEnKS' # chosen method ['AnEnKF', 'AnEnKS', 'AnPF']
fladRegression = 'local_linear'# chosen regression ['locally_constant', 'increment', 'local_linear']
flagSampling   = 'gaussian' # chosen sampler ['gaussian', 'multinomial']
n_EOF          = 0.99
Save           = True
flagPred       = True

# IMPORT DATA######

dataset= np.load("/home/vient/Thèse/Data/Data_ZOI/Dataset_64_ZOI.npy",allow_pickle='TRUE').item()
dataH=dataset['CSED_Hourly']
mask=dataset['Cloud_Daily']
lat_grid=dataset['Lat_ZOI']
lon_grid=dataset['Lon_ZOI']

def prepdata(xH,mask,testsize,N_Catalog=0):
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
    mask_train = mask[:-testsize]
    mask_pred  = mask[-testsize:]
    x_train    = xD[:len(mask_train)]
    y_pred     = xD[len(mask_train):len(mask_train)+len(mask_pred)]
    return mask_train,mask_pred,x_train,y_pred

mask_train,mask_pred,x_train,y_pred = prepdata(dataH,mask,100)

Truepts  = np.where(~np.isnan(x_train[0].reshape(-1)))[0]
Trainsea = x_train.reshape(len(x_train),-1)[:,Truepts]
Predsea  = y_pred.reshape(len(y_pred),-1)[:,Truepts]
masksea = mask_pred.reshape(len(mask_pred),-1)[:,Truepts]

print('Size of train dataset :',Trainsea.shape[0],'---- Prediction on ',len(Predsea),' days')


pca       = PCA(n_EOF)
Cat       = pca.fit_transform(Trainsea)
TransH    = pca.components_
TransMean = pca.mean_
Obs       = Predsea - TransMean + masksea
Obs[0]=Predsea[0]-TransMean
if flagPred:
    Obs[15:,:]= np.nan

class catalog:
    analogs = Cat[0:-1,:]
    successors = Cat[1:,:]

class yo:
    values = Obs
    PCAmean=TransMean
    #time = np.arange(Trainsea.shape[0],len(Trainsea)+len(Predsea))
    time = np.arange(len(y_pred))
size=len(TransH)
# global neighborhood all PCA variable
global_analog_matrix=np.ones((size,size))

# local neighborhood 5 variables
local_analog_matrix=np.eye(size)+np.diag(np.ones(size-1),1)+ np.diag(np.ones(size-1),-1)+ \
                   np.diag(np.ones(size-2),2)+np.diag(np.ones(size-2),-2)+\
                   np.diag(np.ones(size-(size-2)),size-2)+np.diag(np.ones(size-(size-2)),size-2).T+\
                   np.diag(np.ones(size-(size-1)),size-1)+np.diag(np.ones(size-(size-1)),size-1).T


# parameters of the analog forecasting method
class AF:
    k = 250; # number of analogs (nom1000)
    neighborhood = global_analog_matrix # global analogs,global_analog_matrix,local_analog_matrix
    catalog = catalog # catalog with analogs and successors
    regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
    sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')


class DA:
    method = 'AnEnKS' # chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
    N = 1000 # number of members (AnEnKF/AnEnKS) or particles (AnPF) nom(250)
    xb = Cat[-1]
    B = np.cov(Cat.T)
    H = TransH.T
    R = np.full((Predsea.shape[1],Predsea.shape[1]),0.75)#nom(0.75)
    @staticmethod
    def m(x):
        return AnDA_analog_forecasting(x,AF)

print('xb :',DA.xb.shape)
print('B :',DA.B.shape)
print('H :',DA.H.shape)
print('R :',DA.R.shape)
print('cat :',catalog.analogs.shape)
print('yo :',yo.values.shape)

# run the analog data assimilation
x_hat_analog_global = AnDA_data_assimilation(yo, DA)

ui=pca.inverse_transform(x_hat_analog_global.values)
tast=np.copy(y_pred.reshape(len(y_pred),-1))
tast[:,Truepts]=ui
tast[np.where(tast<=-4)]=-4
tast[np.where(tast>=np.nanmax(y_pred))]=np.nanmax(y_pred)
if flagPred :
    print(np.sqrt(np.nanmean((y_pred[15:].flatten()-tast[15:].flatten())**2)))
    if Save:
        np.save(str('AndA_pred'+str(len(y_pred)-15)),tast.reshape(y_pred.shape))
        print('Prediction saved')
else:
    print(np.sqrt(np.nanmean((y_pred.flatten()-tast.flatten())**2)))
    if Save:
        np.save('AndA_pred',tast)
        print('Prediction saved')



'''Rui=[0.1,0.25,0.5,0.75,1,3]
RNe=[100,250,500,750,1000]
Rk=[100,250,500,750,1000]
RMSEtot=dict()
QQ=0

for bite in range(len(Rui)):
    for cul in range(len(RNe)):
        for chier in range(len(Rk)):

            class AF:
                k = Rk[chier]; # number of analogs
                neighborhood = global_analog_matrix # global analogs
                catalog = catalog # catalog with analogs and successors
                regression = 'local_linear' # chosen regression ('locally_constant', 'increment', 'local_linear')
                sampling = 'gaussian' # chosen sampler ('gaussian', 'multinomial')

            class DA:
                method = 'AnEnKS' # chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
                N = RNe[cul] # number of members (AnEnKF/AnEnKS) or particles (AnPF)
                xb = Cat[-1]
                B = np.cov(Cat.T)
                H = TransH.T
                R = np.full((Predsea.shape[1],Predsea.shape[1]),Rui[bite])
                @staticmethod
                def m(x):
                    return AnDA_analog_forecasting(x,AF)

            # run the analog data assimilation
            x_hat_analog_global = AnDA_data_assimilation(yo, DA)

            car=pca.inverse_transform(x_hat_analog_global.values)
            tast=np.copy(y_pred.reshape(100,-1))
            tast[:,Truepts]=car
            tast[np.where(tast<=-4)]=-4
            tast[np.where(tast>=np.nanmax(y_pred))]=np.nanmax(y_pred)
            RMSEtot[QQ]=[np.sqrt(np.nanmean((y_pred.flatten()-tast.flatten())**2)),RNe[cul],Rui[bite],Rk[chier]]
            QQ=QQ+1
            print(np.sqrt(np.nanmean((y_pred.flatten()-tast.flatten())**2)))'''

# RMSE=np.empty((len(RMSEtot),4))
# for i in range (len(RMSEtot)):
#     RMSE[i]=RMSEtot[i]