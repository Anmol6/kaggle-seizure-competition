import time
import numpy as np
import pre_processing as prep 
from scripts.split_safe import SafeDataFilter
import  pre_processing as prep
import scipy.io
import pdb
import matplotlib.pyplot as plt
from matplotlib import cm


if __name__ == "__main__":
    data_length_sec = 600
    sampling_frequency = 400
    nfreq_bands = 12    # can play around with these:
    win_length_sec = 4 
    stride_sec = 2
    features = "meanlog_std"  # will create a new additional bin of standard deviation of other bins

    
    f = scipy.io.loadmat('data/train_1/1_1_0.mat')['dataStruct'][0][0][0]
    #f = np.load('data/train_1_npy/1_1_0.npy')['data'][()]
    #f = np.load('data/test_1_npy/safe/1_1182.npy')['data'][()] Broken
    #f = np.load('data/test_1_npy/safe/1_107.npy')['data'][()]
    #f = np.load('data/train_2_npy/2_539_0.npy')['data'][()]
    print(f.shape)

#    pdb.set_trace()
    # compute_fft accepts a matrix of channels x time, so we gotta transpose
    x = f.T  
    #plt.plot(x[1][::100])
    #plt.show()   
    # Test one observation
    st = time.clock()
    filtered = prep.filter_opt(x, 400, 600, 0.1, 180.0)
    end = time.clock()
    print("PrepFilt: " + str(end-st))
    
    
    st = time.clock()
    new_x = prep.compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
    end = time.clock()
    print("PrepComp: " + str(end-st))


    #pdb.set_trace()
    print(new_x.shape)
    #print new_x
    print(new_x[0])
    img2 = plt.imshow(new_x[0][0:-1],interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
    plt.show()
    img2 = plt.imshow(new_x[1][0:-1],interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
    plt.show()
    

