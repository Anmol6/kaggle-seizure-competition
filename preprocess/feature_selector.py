import os
import sys, getopt
import time
import pickle
import numpy as np
from scripts.split_safe import SafeDataFilter
from scripts.data_scaler import scale_across_time 
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
        
        dataFiles = [x for x in os.listdir(direc) if x.endswith('.p')]
        
        n = len(dataFiles)

        X = np.zeros((n, 10, 16, 10))
        y = np.empty((n, 1),dtype='uint8')
        
        for i, filename in enumerate(dataFiles):
            d = pickle.load( open(os.path.join(direc, filename)) )
            
            # concatenate all features -> (timesteps,channels,features)
            x = np.array([v.as_matrix().T for v in d.values()]).T
          
            X[i,] = x
            
            label = safesplit.get_label(filename)
            #print(filename + ' ' + label)
            #print(str(i) + '/' + str(n) + '] ' + filename + ' ' + label)
            if label is '1':
                y[i] = 1
            elif label is '0':
                y[i] = 0
    
    return (X, y)


def feat_to_X(direc):
    if os.path.exists(direc):
        #for generating y
        safesplit = SafeDataFilter() 
        
        dataFiles = [x for x in os.listdir(direc) if x.endswith('.p')]
        
        X = np.zeros((len(dataFiles), 10, 16, 10))
        
        for i, filename in enumerate(dataFiles):
            d = pickle.load( open(os.path.join(direc, filename)) )
            
            # concatenate all features -> (timesteps,channels,features)
            x = np.array([v.as_matrix().T for v in d.values()]).T
            
            '''
            for v in d.values():
                # shape is (timesteps, channels)
                v_n = v.as_matrix()
            '''
            X[i,] = x

    return X

def impute(x, x_test):
    # expects examples,timestep,channel,feature, calculate mean and impute
    timesteps = x.shape[1]
    channels = x.shape[2]
    features = x.shape[3]
    
    flatten_dim = timesteps * channels * features
    
    x = x.reshape(x.shape[0], flatten_dim)
    x_test = x_test.reshape(x_test.shape[0], flatten_dim)
    x_complete = np.vstack((x, x_test))
    
    # loop over timestep*channel*feature, calculate mean and impute
    for i in range(x_complete.shape[1]):
        mean = np.mean(x_complete[:,i][~np.isnan(x_complete[:,i])])
        
        x_test[:,i][np.isnan(x_test[:,i])] = mean
        x[:,i][np.isnan(x[:,i])] = mean

    # reshape back to examples,timesteps,channels,features
    x = x.reshape(x.shape[0], timesteps, channels, features)
    x_test = x_test.reshape(x_test.shape[0], timesteps, channels, features)
    return x, x_test

def fix_data(x, x_test):
    # expects examples,timestep,channel,feature, calculate mean and impute
    
    # impute and then standardize
    # reshape to examples,timesteps*channels*features to impute in each channel/timestep
    x[x == -np.inf] = np.nan 
    x[x == np.inf] = np.nan 
    x_test[x_test == -np.inf] = np.nan 
    x_test[x_test == np.inf] = np.nan 
    
    x, x_test = impute(x, x_test)

    # standardize expects examples, channels, features, timesteps
    x = np.transpose(x, axes=(0, 2, 3, 1))
    x_test = np.transpose(x_test, axes=(0, 2, 3, 1))
    
    n_examples = x.shape[0]
    n_examples_t = x_test.shape[0]
    n_channels = x.shape[1]
    n_fbins = x.shape[2]
    n_timesteps = x.shape[3]

    # now standardize
    #print(x[1,0,0,:]) 
    _, scalers = scale_across_time(x, x_test=x_test)
    #print(x[1,0,0,:]) 

    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins)) 
        xi = scalers[i].transform(xi)
        xi = xi.reshape((n_examples, n_timesteps, n_fbins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x[:, i, :, :] = xi
   
    for i in range(n_channels):
        xi = np.transpose(x_test[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples_t * n_timesteps, n_fbins)) 
        
        xi = scalers[i].transform(xi)

        xi = xi.reshape((n_examples_t, n_timesteps, n_fbins))
        xi = np.transpose(xi, axes=(0, 2, 1))
        x_test[:, i, :, :] = xi
    
    # transpose it back to before
     
    # returns examples,timestep,channel,feature
    x = np.transpose(x, axes=(0, 3, 1, 2))
    x_test = np.transpose(x_test, axes=(0, 3, 1, 2))   
    return x, x_test, scalers

def save_fixed_data():
    direc = 'data/stats'
    for i in range(3):
        patient = i+1
        testnewdir = os.path.join(direc, 'test_' + str(patient) + '_new')
        traindir = os.path.join(direc, 'train_' + str(patient) + '_npy')
        testolddir = os.path.join(direc, 'test_' + str(patient) + '_npy')
        
        X_o, y_o = feat_to_X_y( traindir )
        X_n, y_n = feat_to_X_y( testolddir )
        X_train = np.concatenate((X_o, X_n), axis = 0)
        y_train = np.concatenate((y_o, y_n), axis = 0)
        # should be examples, timesteps, channels, feat
        X_test = feat_to_X( testnewdir ) 
        
        print(X_train.shape) 
        print(X_test[3,:,5,0]) 

        X_train, X_test, scalers = fix_data(X_train, X_test)
        print(X_train.shape) 
        print(X_test[3,:,5,0]) 
        
        np.save(os.path.join(traindir, 'X_train.npy'), X_train)
        np.save(os.path.join(traindir, 'y_train.npy'), y_train)
        np.save(os.path.join(testnewdir, 'X_test.npy'), X_test)
        pickle.dump(scalers, open(os.path.join(traindir,'scalers.p'), 'wb'))

def save_fixed_ffts():
    direc = 'data/ffts/6band'
    for i in range(3):
        patient = i+1
        testnewdir = os.path.join(direc, 'test_' + str(patient) + '_new')
        traindir = os.path.join(direc, 'train_' + str(patient) + '_npy')
        testolddir = os.path.join(direc, 'test_' + str(patient) + '_npy')
        
        X_o = np.load(os.path.join(traindir, 'X_new.npy'))
        y_o = np.load(os.path.join(traindir, 'y_new.npy'))
        X_n = np.load(os.path.join(testolddir, 'X_new.npy'))
        y_n = np.load(os.path.join(testolddir, 'y_new.npy'))

        X_train = np.concatenate((X_o, X_n), axis = 0)
        del X_n
        del X_o

        y_train = np.concatenate((y_o, y_n), axis = 0)
        del y_n
        del y_o
        
        X_test = np.load(os.path.join(testnewdir, 'X_new_s.npy'))
        
        # convert to examples, timesteps, channels, feat
        X_train = np.swapaxes(X_train, 1, 2)
        X_test = np.swapaxes(X_test, 1, 2)
       

        print(X_train.shape) 
        print(X_test[3,:,5,0]) 

        X_train, X_test, scalers = fix_data(X_train, X_test)
        print(X_train.shape) 
        print(X_test[3,:,5,0]) 
        
        np.save(os.path.join(traindir, 'X_ftrain.npy'), X_train)
        del X_train
        np.save(os.path.join(traindir, 'y_ftrain.npy'), y_train)
        del y_train
        np.save(os.path.join(testnewdir, 'X_ftest_s.npy'), X_test)
        del X_test
        pickle.dump(scalers, open(os.path.join(traindir,'scalers.p'), 'wb'))
        del scalers 

if __name__ == '__main__':
    save_fixed_ffts()
