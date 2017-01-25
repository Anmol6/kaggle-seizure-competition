from scipy.io import loadmat

filep = 'data/train_3'

for i in xrange( len([f for f in os.listdir() if f.endswith('.mat')]) ):
    # starts at 1 rather than 0
    index_u = f_u.index('new_' + num + '_' + str(i+1) + '.mat')
    f_s.append('new_' + num + '_' + str(i+1) + '.mat')
    X_s[i] = X_u[index_u]
