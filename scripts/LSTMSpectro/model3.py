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
from keras.callbacks import Callback, ModelCheckpoint

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics 

save_path = 'models/LSTMSpectro/P3/test4end.h5'

class AUCCheckpoint(Callback):
    def __init__(self, filepath, X_train, y_train):
        self.y_train = y_train
        self.X_train = X_train
        self.filepath = filepath
    def on_train_begin(self, logs={}):
        # don't bother below
        self.best_auc = 0.6 
        self.best_file = ''
        self.aucs = []
        self.aucs_x = []
        self.losses = []
        self.losses_x = []
    
    def on_train_end(self, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses_x.append(logs.get('loss'))
        self.losses.append(logs.get('val_loss'))

        preds = self.model.predict_proba(self.model.validation_data[0])
        roc_auc = metrics.roc_auc_score(self.model.validation_data[1][:,1], preds[:,1])

        preds_x = self.model.predict_proba(X_train[0:600])
        roc_auc_x = metrics.roc_auc_score(y_train[0:600][:,1], preds_x[:,1])
        
        print( 'AUC:  Train: {train:.2f}  Val: {val:.2f}'.format(train=roc_auc_x, val=roc_auc) )
        
        #print( 'AUC Val: {val:.3f}'.format(val=roc_auc) )
        
        if(roc_auc > self.best_auc):
            self.best_auc = roc_auc
            filepath = self.filepath.format(epoch=epoch, val_roc_auc=roc_auc) 
            print( 'Saving best model...' )
            self.model.save(filepath, overwrite=True)
            self.best_file = filepath
        else:
            print( 'No improvement over best...' )
        
        self.aucs.append(roc_auc)
        self.aucs_x.append(roc_auc_x)
        return

    def on_batch_begin(self, batch, logs={}):
        return
         
    def on_batch_end(self, batch, logs={}):
        return

filepath = "models/LSTMSpectro/P3/test4-{epoch:02d}-{val_roc_auc:.3f}.h5"
print('Loading data...')
X_o = np.load('data/ffts/train_3_npy/X_new.npy')
y_o = np.load('data/ffts/train_3_npy/y_new.npy') 
X_n = np.load('data/ffts/test_3_npy/X_new.npy')
y_n = np.load('data/ffts/test_3_npy/y_new.npy')

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

# one hot encode
y_s = np.zeros((y_all.shape[0], 2))
print(y_all)
y_s[:, 1] = (y_all == 1).reshape(y_all.shape[0],)
y_s[:, 0] = (y_all == 0).reshape(y_all.shape[0],)
print(y_s)

pos_weight = np.sum(y_s[:,0])/np.sum(y_s[:,1])
print(pos_weight)

X_train,X_test, y_train, y_test = train_test_split(X_all, y_s, test_size=0.2, stratify=y_all) 
del X_all
print(X_train.shape, 'train sequences')
print(y_train.shape, 'test sequences')
print(X_test.shape, 'test sequences')
print(y_test.shape, 'test sequences')

auc_checkpoint = AUCCheckpoint(filepath, X_train, y_train) 
callbacks_list = [auc_checkpoint]


max_features = 208 
maxlen = 597

# Convolution
filter_length = 3
nb_filter = 256 
#pool_length = 4

# LSTM
lstm_output_size1 = 128 
lstm_output_size2 = 64 

# Training
batch_size = 64 
nb_epoch = 190 

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
model.add(Dropout(0.35))

#model.add(MaxPooling1D(pool_length=pool_length))
model.add(Bidirectional(LSTM(lstm_output_size1, return_sequences=True)))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(lstm_output_size1, return_sequences=True)))
model.add(Dropout(0.20))
model.add(Bidirectional(LSTM(lstm_output_size2, return_sequences=False)))
model.add(Dropout(0.10))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
                metrics=['accuracy'])

'''
print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
'''

print('Train...')
model.fit(X_train, y_train,class_weight={0: 1.0, 1: pos_weight}, batch_size=batch_size
        , nb_epoch=nb_epoch, callbacks=callbacks_list,
          validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

model.save(save_path)

preds = model.predict_proba(X_test)
print(preds)
roc_auc = metrics.roc_auc_score(y_test[:,1], preds[:,1])
print('ROC AUC:', roc_auc)

fpr, tpr, thresholds = metrics.roc_curve(y_test[:,1], preds[:,1])
print(thresholds)
plt.figure(1)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.figure(2)
plt.plot(auc_checkpoint.aucs, label='Val', color='navy')
plt.plot(auc_checkpoint.aucs_x, label='Train', color='black')
plt.legend(loc="lower right")
plt.ylabel('AUC')
plt.xlabel('Epoch')

plt.figure(3)
plt.plot(auc_checkpoint.losses, label='Val', color='navy')
plt.plot(auc_checkpoint.losses_x, label='Train', color='black')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc="lower right")
plt.show()

