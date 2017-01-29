import numpy as np
import scipy as sp
import os
import matplotlib.pylab as plt
from sys import getsizeof
from pandas import DataFrame
import scipy.signal
from split_safe import SafeDataFilter
from sklearn.preprocessing import StandardScaler
from data_scaler import scale_across_features, scale_across_time

list_of_file_paths = []
X_train1 = np.load(list_of_file_paths[0])
X_train2 = np.load(list_of_file_paths[1])
X_train3 = np.load(list_of_file_paths[2])
y_train1 = np.load(list_of_file_paths[3])
y_train2 = np.load(list_of_file_paths[4])
y_train3 = np.load(list_of_file_paths[5])

X_test1 = np.load(list_of_file_paths[6])
X_test2 = np.load(list_of_file_paths[7])
X_test3 = np.load(list_of_file_paths[8])



X_train1[X_train1 < -100000] = 0
X_train2[X_train2 < -100000] = 0
X_train3[X_train3 < -100000] = 0

X_test1[X_test1 < -100000] = 0
X_test2[X_test2 < -100000] = 0
X_test3[X_test3 < -100000] = 0

X_train1 = np.abs(X_train1)
X_train2 = np.abs(X_train2)
X_train3 = np.abs(X_train3)

X_test1 = np.abs(X_test1)
X_test2 = np.abs(X_test2)
X_test3 = np.abs(X_test3)

X_trains = [X_train1, X_train2, X_train3]
X_tests = [X_test1, X_test2, X_test3]
y_trains_old = [y_train1, y_train2, y_train3]
y_trains = [y_train1, y_train2, y_train3]
pos_weight = []

# Accepts X's of shape (examples, channels, time-windows, bins)
# X_trains and X_tests must be list
def scale_reshape(X_trains, X_tests):
    # applying scale_across_time
    for i, X in enumerate(X_trains):
        X_train = X.swapaxes(2, 3)
        X_test = X_tests[i]
        X_test = X_test.swapaxes(2, 3)
        X_trains[i], scalers = scale_across_time(X_train, X_test)
        X_tests[i], additional_scalers = scale_across_time(X_test, scalers=scalers)
    # swaping the axis back and reshaping the features
    for i, X in enumerate(X_trains):
        X_trains[i] = X.swapaxes(2, 3)
        X_trains[i] = X.reshape((X.shape[0], 10, 112))
        X = X_tests[i]
        X_tests[i] = X.swapaxes(2, 3)
        X_tests[i] = X.reshape((X.shape[0], 10, 112))
    return X_trains, X_tests, scalers

X_trains, X_tests, scalers = scale_reshape(X_trains, X_tests)
# Computing y's one-hot vectors and pos_weight
for i, X in enumerate(X_trains):
    ys = np.zeros((y_trains[i].shape[0], 2))
    ys[:, 1] = (y_trains[i] > 0).reshape(y_trains[i].shape[0],)
    ys[:, 0] = (y_trains[i] < 1).reshape(y_trains[i].shape[0],)
    y_trains[i] = ys
    pos_weight.append(np.sum(y_trains[i][:,0])/np.sum(y_trains[i][:,1]))


# Checking:
print(np.nonzero(X_trains[0][X_trains[0] > 50]))
print(np.nonzero(X_trains[0][X_trains[0] < -5]))
print("final: ", X_trains[0].shape)
print("final: ", X_tests[0].shape)
print("final: ", y_trains[0].shape)



#from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution1D
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import roc_auc_score
from numpy.random import randint, random
from keras.regularizers import l2, activity_l2
#from keras.callbacks import EarlyStopping
from sklearn.cross_validation import StratifiedKFold
#from scikit_learn_custom import KerasClassifier

# create your model using this function
def create_model(nb_filter1=16, nb_filter2=32, activation1='relu', l2_weight1=0.0, l2_weight2=0.0,
                 dropout_rate=0.3, optimizer='adam', nb_epoch=10, hidden_dims=112):
    #batch_size = 100
    filter_length = 1
    #n = X_train.shape[0]

    print('Building individual model...')
    model = Sequential()
    model.add(Convolution1D(nb_filter=nb_filter1,
                            filter_length=filter_length,
                            init='glorot_normal',
                            border_mode='valid',
                            activation=activation1,
                            subsample_length=1,
                            W_regularizer=l2(l2_weight1),
                            input_shape=(10, 112)))

    model.add(Convolution1D(nb_filter=nb_filter2,
                            filter_length=1,
                            init='glorot_normal',
                            border_mode='valid',
                            subsample_length=1,
                            W_regularizer=l2(l2_weight2),
                            activation=activation1))

    model.add(Reshape((nb_filter2*10,)))
    model.add(Dense(hidden_dims))
    model.add(Activation(activation1))
    model.add(Dropout(dropout_rate))
    model.add(Dense(hidden_dims))
    model.add(Activation(activation1))
    model.add(Dropout(dropout_rate))
    #model.add(GlobalPoolLayer())
    #model.add(Dense(hidden_dims))
    #model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    #model.summary()
    return model



model = KerasClassifier(build_fn=create_model, batch_size=32, verbose=0)
param_grid = {'nb_filter1': [16, 32],
              'nb_filter2': [32, 48],
              'activation1': ['relu'],
              'dropout_rate': [0.3, 0.5],
              'optimizer': ['adam'],
              'nb_epoch': [10, 20]}

param_distributions = {'nb_filter1': randint(8, 32, 500),
                       'nb_filter2': randint(8, 64, 500),
                       'activation1': ['relu'],
                       'dropout_rate': sp.stats.uniform(0.05, 0.45),
                       'optimizer': ['adam', 'rmsprop', 'nadam'],
                       'nb_epoch': randint(10, 55, 500),
                       'l2_weight1': list(np.linspace(0, 3.5, 100)),
                       'l2_weight2': list(np.linspace(0, 3.5, 100)),
                       'hidden_dims': randint(8, 64, 500)}


n_iter_search = 140

#callbacks = [EarlyStopping(monitor='roc_auc', min_delta=0.0005, patience=9, verbose=0, mode='max')]
#             AUCCheckpoint()]
#s = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=2, verbose=0, mode='auto')
#callbacks = [s]
ind = 1
grid_results = []
for i in range(3):
    fit_params = {'class_weight': {0: 1.0, 1: pos_weight[i]}}
    grid = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, verbose=0, n_jobs=1,
                          scoring='roc_auc', fit_params=fit_params, n_iter=n_iter_search, cv=5)
    X = X_trains[i]
    Y = y_trains[i]
    grid_result = grid.fit(X, Y)
    best_model = grid_result.best_estimator_.model
    best_model.save('Model%d' % i)
    grid_results.append(grid_result)
    print("Model#%d built." % ind)
    ind += 1

    # serialize model to JSON
    model_json = best_model.to_json()
    with open("model_%d.json" % i, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    best_model.save_weights("model_weights_%d.h5" % i)
    print("Saved model to disk")


# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_)))
