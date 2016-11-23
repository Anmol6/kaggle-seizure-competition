import traceback
import errno
from multiprocessing import Pool
import numpy as np
import pre_processing as prep 
from scripts.split_safe import SafeDataFilter
import  pre_processing as prep
import scipy.io
import pdb
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import sys, getopt
import time
import pickle

def compute_X_Y_opt(direc, outdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    safesplit = SafeDataFilter() 
   
    # all files ending in mat or npy
    dataFiles = [x for x in os.listdir(direc) if x.endswith('.npy') or x.endswith('.mat')]
    n = len( dataFiles )
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    X = np.zeros((n, 16, n_timesteps, n_fbins))
    y = np.empty((n, 1),dtype='uint8')
    for i, filename in enumerate(dataFiles):
        if filename.endswith('.mat'):
            f = scipy.io.loadmat(os.path.join(direc, filename))['dataStruct'][0][0][0]
            f = f.T
            filtered = prep.filter_opt(f, sampling_frequency, data_length_sec, 0.1, 180.0)
            new_x = prep.compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = np.swapaxes(new_x,1,2)

            label = safesplit.get_label(filename)
            print(filename + ' ' + label)
            if label is '1':
                y[i] = 1
            elif label is '0':
                y[i] = 0
            
            continue
        elif filename.endswith('.npy'):
            f = np.load(os.path.join(direc, filename))
            if f.shape == (): 
                f = f['data'][()]

            f = f.T
            filtered = prep.filter_opt(f, sampling_frequency, data_length_sec, 0.1, 180.0)
            new_x = prep.compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = np.swapaxes(new_x,1,2)

            label = safesplit.get_label(filename)
            print(str(i) + '/' + str(n) + '] ' + filename + ' ' + label)
            if label is '1':
                y[i] = 1
            elif label is '0':
                y[i] = 0
            
            continue

        else:
            print(filename)
            continue
    np.save(os.path.join(outdir, 'X_new.npy'), X) 
    np.save(os.path.join(outdir, 'y_new.npy'), y) 

def compute_X_unsafe_opt(direc, outdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    dataFiles = [x for x in os.listdir(direc) if x.endswith('.npy') or x.endswith('.mat')]
    n = len( dataFiles )
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    X = np.zeros((n, 16, n_timesteps, n_fbins))
    for i, filename in enumerate(dataFiles):
        print(str(i) + ') ' + filename)
        if filename.endswith('.mat'):
            f = scipy.io.loadmat(os.path.join(direc, filename))['dataStruct'][0][0][0]
            f = f.T
            filtered = prep.filter_opt(f, sampling_frequency, data_length_sec, 0.1, 180.0)
            new_x = prep.compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = np.swapaxes(new_x,1,2)
            continue
        elif filename.endswith('.npy'):
            f = np.load(os.path.join(direc, filename))
            if f.shape == ():  
                f = f['data'][()]
            f = f.T
            filtered = prep.filter_opt(f, sampling_frequency, data_length_sec, 0.1, 180.0)
            new_x = prep.compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = np.swapaxes(new_x,1,2)
            continue
        else:
            continue
    np.save(os.path.join(outdir, 'X_unsafe.npy'), X) 

def compute_X_test_new(direc, outdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    dataFiles = [x for x in os.listdir(direc) if x.endswith('.npy') or x.endswith('.mat')]
    n = len( dataFiles )
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    X = np.zeros((n, 16, n_timesteps, n_fbins))

    for i, filename in enumerate(dataFiles):
        print(str(i) + ') ' + filename)
        if filename.endswith('.mat'):
            f = scipy.io.loadmat(os.path.join(direc, filename))['dataStruct'][0][0][0]
            f = f.T
            filtered = prep.filter_opt(f, sampling_frequency, data_length_sec, 0.1, 180.0)
            new_x = prep.compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = np.swapaxes(new_x,1,2)
            continue
        elif filename.endswith('.npy'):
            f = np.load(os.path.join(direc, filename))
            if f.shape == ():  
                f = f['data'][()]
            f = f.T
            filtered = prep.filter_opt(f, sampling_frequency, data_length_sec, 0.1, 180.0)
            new_x = prep.compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = np.swapaxes(new_x,1,2)
            continue
        else:
            continue
    np.save(os.path.join(outdir, 'X_new.npy'), X) 
    pickle.dump(dataFiles, open(os.path.join(outdir,'filenames.p'), 'wb')) 

def flush():
    inputdir = 'data'
    outputdir = 'data/ffts'
    #files = ['train_1_npy', 'train_2_npy', 'train_3_npy', 'test_1_npy', 'test_2_npy', 'test_3_npy']
    files = ['test_1_new','test_2_new','test_3_new']
    indirs = [os.path.join(inputdir,x) for x in files]
    outdirs= [os.path.join(outputdir,x) for x in files]
    inouts = zip(indirs,outdirs)

    # create the output directories
    try:
        os.makedirs(outputdir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
   
    for filen in outdirs:
        try:
            os.makedirs(filen)
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
    
    print(inouts)
    
    # now assign each thread a file
    p = Pool(len(inouts))
    p.map(doone, inouts)

def doone(inout):
    data_length_sec = 600
    sampling_frequency = 400
    nfreq_bands = 12    # can play around with these:
    win_length_sec = 4 
    stride_sec = 1
    features = "meanlog_std"  # will create a new additional bin of standard deviation of other bins

    print(inout)
    
    try:
        inputdir, outputdir = inout

        #compute_X_unsafe_opt(inputdir, outputdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
        print('safe')
        
        # check if there is a safe folder in input dir
        if os.path.exists(os.path.join(inputdir, 'safe')):
            compute_X_Y_opt(os.path.join(inputdir,'safe'), outputdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
        
        compute_X_test_new(inputdir, outputdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)


    except:
        # Put all exception text into an exception and raise that
        print(inout)
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))

'''
def dooneMulti(inout):
    data_length_sec = 600
    sampling_frequency = 400
    nfreq_bands = 12    # can play around with these:
    win_length_sec = 4 
    stride_sec = 1
    features = "meanlog_std"  # will create a new additional bin of standard deviation of other bins

    print(inout)
    
    try:
        inputdir, outputdir = inout

        #compute_X_unsafe_opt(inputdir, outputdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
        print('safe')
        try:
            os.makedirs(os.path.join(outputdir,'1'))
            os.makedirs(os.path.join(outputdir,'2'))
            os.makedirs(os.path.join(outputdir,'3'))
        except OSError as exception:
            if exception.errno != errno.EEXIST:
                raise
 
        # check if there is a safe folder in input dir
        if os.path.exists(os.path.join(inputdir, 'safe')):
            # split the work in two
            compute_X_Y_opt(os.path.join(inputdir,'safe'), outputdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
    
    except:
        # Put all exception text into an exception and raise that
        print(inout)
        raise Exception("".join(traceback.format_exception(*sys.exc_info())))
'''

if __name__ == "__main__":
    flush()
    '''
    data_length_sec = 600
    sampling_frequency = 400
    nfreq_bands = 6    # can play around with these:
    win_length_sec = 60 
    stride_sec = 60
    features = "meanlog_std"
    
    data_length_sec = 600
    sampling_frequency = 400
    nfreq_bands = 12    # can play around with these:
    win_length_sec = 4 
    stride_sec = 2
    features = "meanlog_std"  # will create a new additional bin of standard deviation of other bins
    

    argv = sys.argv[1:]

    inputdir = ''
    outputdir = ''
    try:
        opts, args = getopt.getopt(argv, "i:o:", ["idir=","odir="])
    except getopt.GetoptError:
        print( 'test.py -i <inputdir> -o <outputdir>' )
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-i", "--idir"):
            inputdir = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
    
    if(not inputdir or not outputdir):
        print( 'test.py -i <inputdir> -o <outputdir>' )
        sys.exit(2)

    direc = inputdir 
    outdir = outputdir 

    compute_X_Y_opt(direc, outdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
    '''
