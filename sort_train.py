import pickle
import numpy as np
import os

in_path = 'data/ffts/6band/train_1_new'
num = '1'

f_u = pickle.load(open(os.path.join(in_path,'filenames.p'), 'rb'))
X_u = np.load(os.path.join(in_path, 'X_ftrain.npy'))
y_u = np.load(os.path.join(in_path, 'y_ftrain.npy'))
f_s = []
y_s = np.zeros(y_u.shape) 
X_s = np.zeros(X_u.shape)

print('y: ', len(f_u))
print('X: ', X_u.shape)
# Increment index and reindex X 
for i in xrange(len(f_u)):
    # starts at 1 rather than 0
    index_u = f_u.index('new_' + num + '_' + str(i+1) + '.mat')
    f_s.append('new_' + num + '_' + str(i+1) + '.mat')
    X_s[i] = X_u[index_u]

print(len(f_s))
print(f_s)
print(X_s)

np.save(os.path.join(in_path, 'X_new_s.npy'), X_s)
pickle.dump(f_s, open(os.path.join(in_path, 'filenames_s.p'), 'wb'))
