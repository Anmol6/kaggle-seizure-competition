import os
import sys, getopt
import time
import pickle
import numpy as np
from scripts.split_safe import SafeDataFilter
import scripts.data_scaler as dscaler
# Import python libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


'''
Goes over raw feature_extractor ouput and generates X and y ndarrays.
'''

def feat_to_X_y(direc):
    if os.path.exists(direc):
        #for generating y
        safesplit = SafeDataFilter() 
        
        dataFiles = [x for x in os.listdir(direc) if x.endswith('.npy') or x.endswith('.mat')]
        
        X = np.zeros((len(dataFiles), 10, 16, 10))
        y = np.empty((n, 1),dtype='uint8')
        
        for i, filename in enumerate(dataFiles):
            d = pickle.load(open('filename'))
            
            # concatenate all features -> (timesteps,channels,features)
            x = np.array([v.as_matrix().T for v in d.values()]).T
          
            X[i,] = x
            
            label = safesplit.get_label(filename)
            print(filename + ' ' + label)
            print(str(i) + '/' + str(n) + '] ' + filename + ' ' + label)
            if label is '1':
                y[i] = 1
            elif label is '0':
                y[i] = 0
    
    return (X, y)


def feat_to_X(direc):
    if os.path.exists(direc):
        #for generating y
        safesplit = SafeDataFilter() 
        
        dataFiles = [x for x in os.listdir(direc) if x.endswith('.npy') or x.endswith('.mat')]
        
        X = np.zeros((len(dataFiles), 10, 16, 10))
        
        for i, filename in enumerate(dataFiles):
            d = pickle.load(open('filename'))
            
            # concatenate all features -> (timesteps,channels,features)
            x = np.array([v.as_matrix().T for v in d.values()]).T
            
            '''
            for v in d.values():
                # shape is (timesteps, channels)
                v_n = v.as_matrix()
            '''
            X[i,] = x
    
    x_f = x.reshape(X_all)  

    return X

def impute(x, x_test):
    # expects examples,timestep,channel,feature, calculate mean and impute
    flatten_dim = x.shape[1]*x.shape[2]*x.shape[3]
    x = x.reshape(x.shape[0], flatten_dim)
    x_test = x_test.reshape(x_test.shape[0], flatten_dim)
    x_complete = np.vstack((x, x_test))
    
    # loop over timestep*channel*feature, calculate mean and impute
    for i in range(x_complete.shape[1]):
        mean = np.mean(x_complete[:,i][~np.isnan(x_complete[:,i])])
        
        x_test[:,i](np.isnan(x_test[:,i])) = mean
        x[:,i](np.isnan(x[:,i])) = mean

    # reshape back to examples,timesteps,channels,features
    x = x.reshape(examples, timesteps, channels, features)
    x_test = x_test.reshape(examples, timesteps, channels, features)
    return x, x_test

def fix_data(x, x_test)
    # impute and then standardize
    # reshape to examples,timesteps*channels*features to impute in each channel/timestep
    x, x_test = impute(x, x_test)

    # standardize expects examples, channels, features, timesteps
    x = np.transpose(x, axes=(0, 2, 3, 1))
    x_test = np.transpose(x_test, axes=(0, 2, 3, 1))

    # now standardize
    _, scalers = scale_across_time(x, x_test=x_test)

    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins)) 
        
        xi = scalers[i].transform(xi)

        xi = xi.reshape((n_examples, n_timesteps, n_fbins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x[:, i, :, :] = xi
   
    for i in range(n_channels):
        xi = np.transpose(x_test[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins)) 
        
        xi = scalers[i].transform(xi)

        xi = xi.reshape((n_examples, n_timesteps, n_fbins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x_test[:, i, :, :] = xi
    
    # transpose it back to before
    # save x and x_test
    

    return x_train, x_test, scalers
