import numpy as np
import pre_processing as prep 
from scripts.split_safe import SafeDataFilter
import  pre_processing as prep
import scipy.io
import pdb

def compute_X_Y_opt(direc, outdir, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    safesplit = SafeDataFilter() 
    
    n = len([name for name in os.listdir(direc)])
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    X = np.zeros((n, 16, n_timesteps, n_fbins))
    y = np.empty((n, 1))
    for i, filename in enumerate(os.listdir(direc)):
        if filename.endswith('.mat'):
            f = scipy.io.loadmat(direc + filename)['dataStruct'][0][0][0]
            f = f.T
            filtered = prep.filter_opt(f, sampling_frequency, datalength_sec, 0.1, 180.0)
            new_x = compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = new_x

            label = safesplit.get_label(filename)
            if label is '1':
                y[i] = 1
            elif label is '0':
                y[i] = 0
            
            continue
        else:
            continue
    numpy.save(X, os.path.join(outdir, 'X_new.npy')) 
    numpy.save(y, os.path.join(outdir, 'y_new.npy')) 

if __name__ == "__main__":
    data_length_sec = 600
    sampling_frequency = 400
    nfreq_bands = 12    # can play around with these:
    win_length_sec = 25 
    stride_sec = 1
    features = "meanlog_std"  # will create a new additional bin of standard deviation of other bins


    f = scipy.io.loadmat('data/train_1/1_1_0.mat')['dataStruct'][0][0][0]
#    pdb.set_trace()
    # compute_fft accepts a matrix of channels x time, so we gotta transpose
    x = f.T  
    
    # Test one observation
    filtered = prep.filter_opt(x, 400, 600, 0.1, 180.0)
    new_x = prep.compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
    #pdb.set_trace()
    print(new_x.shape)
    #print new_x
    print(new_x[0])
    img2 = plt.imshow(new_x[0][0:-1],interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
    plt.show()
    img2 = plt.imshow(new_x[1][0:-1],interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
    plt.show()
    

