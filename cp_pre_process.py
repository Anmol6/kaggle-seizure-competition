import pre_processing

def compute_X_Y_opt(direc, data_length_sec, sampling_frequency, nfreq_bands, win_length_sec, stride_sec, features):
    n = len([name for name in os.listdir(direc)])
    n_timesteps = (data_length_sec - win_length_sec) / stride_sec + 1
    n_fbins = nfreq_bands + 1 if 'std' in features else nfreq_bands

    X = np.zeros((n, 16, n_timesteps, n_fbins))
    y = np.empty((n, 1))
    for i, filename in enumerate(os.listdir(direc)):
        if filename.endswith('.npy'):
            f = np.load(direc + filename)
            f = f.T
            filtered = filter_freq(f, sampling_frequency, datalength_sec, 0.1, 180.0)
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
    

