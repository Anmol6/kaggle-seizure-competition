from numpy import vstack
from sklearn.decomposition import KernelPCA 
import numpy as np
import gc
import pickle
import os
    
kpca_params = {'n_components':200,
               'kernel':'rbf',
               'gamma':None,
               'degree':3,
               'coef0':1.6,
               'kernel_params':None,
               'alpha':1.0,
               'fit_inverse_transform':False,
               'eigen_solver':'auto',
               'tol':0,
               'max_iter':None,
               'remove_zero_eig':True,
               'n_jobs':7}

folder = 'data/stats/'

for i in range(3):
    patient = i+1
    
    X_all = np.load(folder + 'train_' + str(patient)+ '_npy/X_train.npy')
    X_sub = np.load(folder + 'test_' + str(patient)+ '_new/X_test.npy') 

    X_all = X_all.reshape(X_all.shape[0],X_all.shape[1]*X_all.shape[2]*X_all.shape[3])
    X_sub = X_sub.reshape(X_sub.shape[0],X_sub.shape[1]*X_sub.shape[2]*X_sub.shape[3])
    X = vstack((X_sub, X_all))
    del X_all, X_sub; gc.collect()

    kpca = KernelPCA(**kpca_params)
    kpca.fit(X)
    with open(os.path.join(folder+'train_' + str(patient)+ '_npy', 'kpca.p'),'wb') as f:
        pickle.dump(kpca,f)
    del X, kpca; gc.collect()

