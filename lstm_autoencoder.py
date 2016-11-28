import numpy as np
import scipy.io as sio
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras import backend as K
from keras.optimizers import Adam
import preprocessing as prep 



#load the .mat file, returns (240000,16) array

def load_file(filename):
	return sio.loadmat(filename)['dataStruct'][0][0][0]


def filter_fft(new_x):
	new_x[new_x < 0] = 0
	return new_x


#f1 = np.array(load_file('2_1_0.mat'))
#f1 = np.reshape(f1, (240000,1,16))

#f2 = np.array(load_file('2_1_1.mat'))
#f2 = np.reshape(f2, (240000,1,16))
#in_dim = f1.shape[1:]






#def main():


#testfiles = prep.compute_X_Y('/media/anmol/My Passport/kaggle/ffts/test_1_new','p1')
#np.save(testfiles, 'test_1_new_1.npy')
f1 = np.load('/media/anmol/My Passport/kaggle/ffts/train_3_npy/X_unsafe.npy')
f1 = filter_fft(f1)
#f1 = testfiles[0]
f1 = np.reshape(f1, (f1.shape[0], f1.shape[2] , f1.shape[1] * f1.shape[3]))

print f1  
LSTM_dim = 150
in_dim = f1.shape[2]

print f1.shape
print "im here 1"

a1 = (f1==-np.inf)
print (np.sum(a1))

m = Sequential()
m.add(LSTM(LSTM_dim, input_dim = in_dim, return_sequences=True))
#m.add(LSTM(50))
print "im here 2"
m.add(LSTM(in_dim, return_sequences=True))
#adam = Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-09, decay=0.0)

m.compile(loss='mean_squared_error', optimizer='adam')
#m.load_weights('lstm_autoencoder.h5')
print "im here 3"
m.load_weights('lstm_autoencoder_p3.h5')
m.fit(f1,f1, batch_size = 30, nb_epoch = 150)
m.save_weights('lstm_autoencoder_p3.h5')

m.load_weights('lstm_autoencoder_p3.h5')
m.fit(f1,f1, batch_size = 30, nb_epoch = 150)
m.save_weights('lstm_autoencoder_p3.h5')

#get_features = K.function([m.layers[0].input], [m.layers[0].output])
#out = get_features([f1])[0]

#	return;

#if __name__== "main":
#	main();
