import os
import sys, getopt
import time
import pickle
import numpy as np
from scripts.split_safe import SafeDataFilter

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
    return X
