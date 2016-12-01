import os
import pandas as pd
import numpy as np

sample_submission = pd.read_csv('sample_submission.csv')
print(sample_submission['Class'])
summ = 0


preds1 = np.load('preds1.npy')
summ+=np.sum(preds1>0.4)

preds2 = np.load('preds2.npy')
summ+=np.sum(preds2>0.4)
preds3 = np.load('preds3.npy')
summ+=np.sum(preds3>0.4)


print summ

preds_submission = np.concatenate((preds1,preds2,preds3))
print(preds_submission)
print(preds_submission.shape)
print "here1"
sample_submission['Class'] = preds_submission
print "here2"
sample_submission.to_csv('tosubmit.csv', index=False)
