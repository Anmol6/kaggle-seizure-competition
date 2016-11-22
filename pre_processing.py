import numpy as np
from functools import partial
#from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib import cm
from pandas import DataFrame
import scipy as sc
import scipy.signal
import os
import pdb
import time

def group_into_bands(fft, fft_freq, nfreq_bands):
    if nfreq_bands == 178:
        bands = range(1, 180)
    elif nfreq_bands == 4:
        bands = [0.1, 4, 8, 12, 30]
    elif nfreq_bands == 6:
        bands = [0.1, 4, 8, 12, 30, 70, 180]
    # http://onlinelibrary.wiley.com/doi/10.1111/j.1528-1167.2011.03138.x/pdf
    elif nfreq_bands == 8:
        bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
    elif nfreq_bands == 12:
        bands = [0.5, 4, 8, 12, 30, 40, 50, 60, 70, 85, 100, 140, 180]
    elif nfreq_bands == 9:
        bands = [0.1, 4, 8, 12, 21, 30, 50, 70, 100, 180]
    else:
        raise ValueError('wrong number of frequency bands')
    
    freq_bands = np.digitize(fft_freq, bands)
    #print(freq_bands)
    #st = time.clock()
    df = DataFrame({'fft': fft, 'band': freq_bands})
    #end = time.clock()
    #print("DF: " + str(end-st))


    #st = time.clock()
    df = df.groupby('band').mean()
    #end = time.clock()
    #print("DFM: " + str(end-st))


    return df.fft[1:-1]

def fgroup_into_bands(fft, fft_freq, nfreq_bands):
    if nfreq_bands == 178:
        bands = range(1, 180)
    elif nfreq_bands == 4:
        bands = [0.1, 4, 8, 12, 30]
    elif nfreq_bands == 6:
        bands = [0.1, 4, 8, 12, 30, 70, 180]
    # http://onlinelibrary.wiley.com/doi/10.1111/j.1528-1167.2011.03138.x/pdf
    elif nfreq_bands == 8:
        bands = [0.1, 4, 8, 12, 30, 50, 70, 100, 180]
    elif nfreq_bands == 12:
        bands = [0.5, 4, 8, 12, 30, 40, 50, 60, 70, 85, 100, 140, 180]
    elif nfreq_bands == 9:
        bands = [0.1, 4, 8, 12, 21, 30, 50, 70, 100, 180]
    else:
        raise ValueError('wrong number of frequency bands')
    freq_bands = np.digitize(fft_freq, bands)
    cutoff_index = [0]
    
    for n in xrange(freq_bands.size):
        if(freq_bands[n] != freq_bands[cutoff_index[-1]]):
            cutoff_index.append(n)
    
    # the last case is special since it goes to the end
    # also we dont need the first bin since we disregard frequencies below lowest bin
    df=np.zeros(nfreq_bands)
    for n in xrange(2,len(cutoff_index)-1):
        df[n-2] = np.mean(fft[cutoff_index[n-1]:cutoff_index[n]]) 
    df[-1] = np.mean(fft[cutoff_index[-2]:cutoff_index[-1]])
    # we assume that fft_freq is only increasing
    #df = DataFrame({'fft': fft, 'band': freq_bands})
    #df = df.groupby('band').mean()
    
    return df

# returns channels x bins x time-frames
def compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    n_channels = x.shape[0]
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    x2 = np.zeros((n_channels, n_fbins, n_timesteps))
    count = 0
    for i in range(n_channels):
        xc = np.zeros((n_fbins, n_timesteps))
        for frame_num, w in enumerate(range(0, data_length_sec - win_length_sec + 1, stride_sec)):
            count += 1
            #print frame_num, w
            xw = x[i, w * sampling_frequency: (w + win_length_sec) * sampling_frequency]

            st = time.clock()
            fft = np.log10(np.absolute(np.fft.rfft(xw, axis = -1)))
            end = time.clock()
            if(count % 111 == 0): print("LOGFFT: " + str(end-st))

            st = time.clock()
            fft_freq = np.fft.rfftfreq(n=xw.shape[-1], d=1.0 / sampling_frequency)
            fft = np.log10(np.absolute(np.fft.rfft(xw, axis = -1)))
            end = time.clock()
            if(count % 111 == 0): print("RFFTFREQ: " + str(end-st))

            
            #if(frame_num == 1): print(fft_freq)
            '''  
            st = time.clock()
            if(count % 111 == 0):print(group_into_bands(fft, fft_freq, nfreq_bands))
            end = time.clock()
            if(count % 111 == 0): print("GroupBands: " + str(end-st))
           
            st = time.clock()
            if(count % 111 == 0):print(fgroup_into_bands(fft, fft_freq, nfreq_bands))
            end = time.clock()
            if(count % 111 == 0): print("FGroupBands: " + str(end-st))
            '''

            st = time.clock()
            xc[:nfreq_bands, frame_num] = fgroup_into_bands(fft, fft_freq, nfreq_bands)
            end = time.clock()
            if(count % 111 == 0): print("FGroupBands: " + str(end-st))

            st = time.clock()
            if 'std' in features:
                xc[-1, frame_num] = np.std(xw)
            end = time.clock()
            if(count % 111 == 0): print("STD: " + str(end-st))
        st = time.clock()
        x2[i, :, :] = xc
        end = time.clock()
        print("ASSIGN: " + str(end-st))
  
    print(count)
    print(np.amax(x2))
    print(np.amin(x2))
    return x2

# filters out the low freq and high freq 
def filter_opt(x, new_sampling_frequency, data_length_sec, lowcut, highcut):
    st = time.clock()
    x1 = scipy.signal.resample(x, new_sampling_frequency * data_length_sec, axis=1)
    end = time.clock()
    print("Resample: " + str(end-st))
    nyq = 0.5 * new_sampling_frequency
    
    st = time.clock()
    b, a = sc.signal.butter(5, np.array([lowcut, highcut]) / nyq, btype='band')
    end = time.clock()
    print("Butter: " + str(end-st))
    
    st = time.clock()
    x_filt = sc.signal.lfilter(b, a, x1, axis=1)
    end = time.clock()
    print("lFilter: " + str(end-st))
    return np.float32(x_filt)
# Computes X and y from all the .npy files in a directory
# X = n x channels x filters x time-frames
# y = n x 1
def compute_X_Y(direc):
    n = len([name for name in os.listdir(direc)])
    X = np.zeros((n, 16, 7, 10))
    y = np.empty((n, 1))
    for i, filename in enumerate(os.listdir(direc)):
        if filename.endswith('.npy'):
            f = np.load(direc + filename)
            f = f.T
            filtered = filter_freq(f, 400, 600, 0.1, 180.0)
            new_x = compute_fft(filtered, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
            X[i, ] = new_x
            if filename.endswith('1.npy'):
                y[i] = 1
            elif filename.endswith('0.npy'):
                y[i] = 0
            continue
        else:
            continue
    
    return X, y

'''
direc_train = '/Users/Anuar_The_Great/desktop/ML/train_1_npy_train/'
direc_test = '/Users/Anuar_The_Great/desktop/ML/train_1_npy_test/'
X_train, y_train = compute_X_Y(direc_train)
X_test, y_test = compute_X_Y(direc_test)
'''
if __name__ == "__main__":
    data_length_sec = 600
    sampling_frequency = 400
    nfreq_bands = 12    # can play around with these:
    win_length_sec = 25 
    stride_sec = 1
    features = "meanlog_std"  # will create a new additional bin of standard deviation of other bins


    f = np.load('data/train_1_npy/1_1_0.npy')['data'][()]
    # compute_fft accepts a matrix of channels x time, so we gotta transpose
    x = f.T  
    
    # Test one observation
    filtered = filter(x, 400, 600, 0.1, 180.0)
    new_x = compute_fft(x, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features)
    #pdb.set_trace()
    print(new_x.shape)
    #print new_x
    print(new_x[0])
    img2 = plt.imshow(new_x[0][0:-1],interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
    plt.show()
    img2 = plt.imshow(new_x[1][0:-1],interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
    plt.show()
