import os
import pandas as pd
import numpy as np
from keras.models import load_model

sample_submission = pd.read_csv('submissions/sample_submission.csv')
print(sample_submission['Class'])

def preproc(X_all):
    X_all[X_all == -np.inf] = -10
    X_all[X_all > 1000] = 1000
    X_all = np.swapaxes(X_all, 1, 2)
    X_all = X_all.reshape(X_all.shape[0],X_all.shape[1],X_all.shape[2]*X_all.shape[3])
    return X_all

model1 = load_model('models/LSTMSpectro/P1/test2.h5')
X_s_1 = np.load('data/ffts/test_1_new/X_new_s.npy')
X_s_1 = preproc(X_s_1)
preds1 = model1.predict_proba(X_s_1)[:,1]
print(preds1)
print(preds1.shape)
del X_s_1
del model1

model2 = load_model('models/LSTMSpectro/P2/test3.hf5')
X_s_2 = np.load('data/ffts/test_2_new/X_new_s.npy')
X_s_2 = preproc(X_s_2)
preds2 = model2.predict_proba(X_s_2)[:,1]
print(preds2)
print(preds2.shape)
del X_s_2
del model2

model3 = load_model('models/LSTMSpectro/P3/test2.h5')
X_s_3 = np.load('data/ffts/test_3_new/X_new_s.npy')
X_s_3 = preproc(X_s_3)
preds3 = model3.predict_proba(X_s_3)[:,1]
print(preds3)
print(preds3.shape)
del X_s_3
del model3

preds_submission = np.concatenate((preds1,preds2,preds3))
print(preds_submission.shape)
sample_submission['Class'] = preds_submission
sample_submission.to_csv('submissions/LSTMSpectro232.csv', index=False)
