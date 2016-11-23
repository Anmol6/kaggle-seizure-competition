from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.layers import Convolution1D, MaxPooling1D

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print('Loading data...')
X_o = np.load('data/ffts/train_1_npy/X_new.npy')
y_o = np.load('data/ffts/train_1_npy/y_new.npy') 
X_n = np.load('data/ffts/test_1_npy/X_new.npy')
y_n = np.load('data/ffts/test_1_npy/y_new.npy')

X_all = np.concatenate((X_o, X_n), axis = 0)
y_all = np.concatenate((y_o, y_n), axis = 0)

# min and max inputs
X_all[X_all == -np.inf] = -10
X_all[X_all > 1000] = 1000

print(np.amax(X_all))
print(np.amin(X_all))
print(np.unravel_index(X_all.argmax(), X_all.shape))

X_all = np.swapaxes(X_all, 1, 2)
'''
img2 = plt.imshow(X_all[0][0:,0,0:-1].T,interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
plt.show()
'''
X_all = X_all.reshape(X_all.shape[0],X_all.shape[1],X_all.shape[2]*X_all.shape[3])
'''
X_all[X_all>4] = 4
img2 = plt.imshow(X_all[782][0:,0:].T,interpolation='nearest', cmap = cm.gist_rainbow, origin='lower')
plt.show()
'''
print(X_all.shape)

max_features = 208 
maxlen = 597

# Convolution
filter_length = 1
nb_filter = 64
#pool_length = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 330
nb_epoch = 5

'''
Note:
batch_size is highly sensitive.
'''
print('Build model...')

model = Sequential()
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1, input_shape=(maxlen,max_features)))
model.add(Dropout(0.20))

#model.add(MaxPooling1D(pool_length=pool_length))
model.add(Bidirectional(LSTM(lstm_output_size, return_sequences=False)))
model.add(Dropout(0.20))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
                metrics=['accuracy'])


X_train,X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, stratify=y_all) 
print(X_train.shape, 'train sequences')
print(y_train.shape, 'test sequences')
print(X_test.shape, 'test sequences')
print(y_test.shape, 'test sequences')

'''
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
'''


print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

preds = model.predict_classes(X_test)
print('ROC AUC:', roc_auc_score(y_test, preds))
