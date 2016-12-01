import os
import random
patient = '1'
l0 = [x for x in os.listdir('./data/stats/train_' + patient + '_npy') if x.endswith('0.p')]
l1 = [x for x in os.listdir('./data/stats/train_' + patient + '_npy') if x.endswith('1.p')]
l1t = [x for x in os.listdir('./data/stats/test_' + patient + '_npy') if x.endswith('.p')]
l0num = [int(x.split('_')[1]) for x in l0]
l1num = [int(x.split('_')[1]) for x in l1]
l1tnum = [int(x.split('_')[1].split('.')[0]) for x in l1t]

l0group = []
for i in range(max(l0num)/6+1):
    lt = []
    for j in range (6):
            id = j+1+6*i
            if id in l0num:
                    lt.append(patient + '_' + str(id) + '_0.p')
    if (lt):
            l0group.append(lt)

l1group=[]
for i in range(max(l1num)/6+1):
    lt = []
    for j in range (6):
            id = j+1+6*i
            if id in l1num:
                    lt.append(patient + '_' + str(id) + '_1.p')
    if (lt):
            l1group.append(lt)

l1tgroup=[]
for i in range(max(l1tnum)/6+1):
    lt = []
    for j in range (6):
            id = j+1+6*i
            if id in l1tnum:
                    lt.append(patient + '_' + str(id) + '.p')
    if (lt):
            l1tgroup.append(lt)

l1tot = l1group + l1tgroup
random.shuffle(l0group)
random.shuffle(l1tot)
len([val for sublist in l1tot[25:] for val in sublist])
len([val for sublist in l0group[25:] for val in sublist])

val= l1tot[0:24] + l0group[0:25]
train= l1tot[24:] + l0group[25:]

val = [x for sublist in val for x in sublist]
train = [x for sublist in train for x in sublist]

import numpy as np

Xold = np.load('data/ffts/train_' + patient + '_npy/X_train.npy')
yold = np.load('data/ffts/train_' + patient + '_npy/y_train.npy')

# assume original order was stacked os listdirs
f_u = [x for x in os.listdir('./data/stats/train_' + patient + '_npy') if x.endswith('0.p') or x.endswith('1.p')] + [x for x in os.listdir('./data/stats/test_' + patient + '_npy') if x.endswith('.p')]

X_val = np.zeros((len(val),Xold.shape[1],Xold.shape[2],Xold.shape[3]))
y_val = np.zeros(len(val))
X_train = np.zeros((len(train),Xold.shape[1],Xold.shape[2],Xold.shape[3]))
y_train = np.zeros(len(train))

f_train = []
for i, t in enumerate(train):
    # starts at 1 rather than 0
    index_u = f_u.index(t)
    f_train.append(t)
    X_train[i] = Xold[index_u]
    y_train[i] = yold[index_u]

f_val = []
for i, t in enumerate(val):
    # starts at 1 rather than 0
    index_u = f_u.index(t)
    f_val.append(t)
    X_val[i] = Xold[index_u]
    y_val[i] = yold[index_u]

# save everything to disc

np.save('data/stats/train_' + patient + '_npy/X_fstrain.npy', X_train)
del X_train
np.save('data/stats/train_' + patient + '_npy/y_fstrain.npy', y_train)
np.save('data/stats/train_' + patient + '_npy/X_fsval.npy', X_val)
np.save('data/stats/train_' + patient + '_npy/y_fsval.npy', y_val)
import pickle
pickle.dump(f_train, open('data/stats/train_' + patient + '_npy/f_fstrain.p', 'wb'))
pickle.dump(f_train, open('data/stats/train_' + patient + '_npy/f_fsval.p', 'wb'))

