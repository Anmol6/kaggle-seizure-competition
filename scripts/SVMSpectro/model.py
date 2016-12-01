#SVM for ffts
import pickle
import os
import numpy as np
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier as bgclf

def sortshit(patient):

	in_path = '/media/anmol/My Passport/kaggle/ffts_10/test_' + str(patient)+'_new'
	num = str(patient)

	f_u = pickle.load(open(os.path.join(in_path,'filenames.p'), 'rb'))
	X_u = np.load(os.path.join(in_path, 'X_new.npy'))
	f_s = []
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
	return X_s
#folder cheng was using 'data/ffts/train_' + str(patient)+ '_npy/X_new.npy'
# fodler anmol is using '/media/anmol/My Passport/kaggle/ffts

def load_data(patient):
	folder = '/media/anmol/My Passport/kaggle/stats1/'
	X_train = np.load(folder + 'train_' + str(patient)+ '_npy/X_fstrain.npy')
	y_train = np.load(folder + 'train_' + str(patient)+ '_npy/y_fstrain.npy') 
	X_test = np.load(folder + 'train_' + str(patient)+ '_npy/X_fsval.npy')
	y_test = np.load(folder + 'train_' + str(patient)+ '_npy/y_fsval.npy') 
	X_sub = np.load(folder + 'test_' + str(patient)+ '_new/X_test.npy') 
	return X_train, y_train, X_test, y_test, X_sub 
#	return X_all, np.reshape(y_all, (y_all.shape[0],))  

patients  = [1,2,3]

preds = np.zeros(3,)

for p in patients:
	

	X_train, y_train, X_test, y_test, X_sub = load_data(p)
	print(X_train.shape)
	X_train = np.swapaxes(X_train, 1, 2)
	X_test = np.swapaxes(X_test, 1, 2)

	X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
	print(X_train.shape)
	X_sub = X_sub.reshape(X_sub.shape[0],X_sub.shape[1]*X_sub.shape[2]*X_sub.shape[3])
	X_test_real = X_sub

	y_s = np.zeros((y_train.shape[0], 2))
	y_s[:, 1] = (y_train == 1).reshape(y_train.shape[0],)
	y_s[:, 0] = (y_train == 0).reshape(y_train.shape[0],)

	pos_weight = np.sum(y_s[:,0])/np.sum(y_s[:,1])	




	#X_train,X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.05, stratify=y_all)

	#weights = (y_train.shape[0]-np.sum(y_train))/np.sum(y_train)
	y_test = y_test.ravel()
	y_train = y_train.ravel()

	print "weights:"
	#print weights



	print(X_train.shape, 'train sequences')
	print(y_train.shape, 'train sequences')
	print(X_test.shape, 'test sequences')
	print(y_test.shape, 'test sequences')
	#del X_all

	#nb_iters = 80000
	svm = SVC(class_weight = {0:1.0, 1: pos_weight }, probability = True)# max_iter =  nb_iters)
	clf = bgclf(svm, max_samples = 0.3)


	clf.fit(X_train, y_train)
	#clf.fit(X_test, y_test)
	preds = clf.predict_proba(X_test)
	print("Predictions:")
	print(preds)
	print("Cross_val score:")

	print(clf.score(X_test,y_test))
	

	roc_auc = metrics.roc_auc_score(y_test, preds[:,1])
	print('ROC AUC:', roc_auc)

	print("REAL Predictions:")

	

	preds_real = clf.predict_proba(X_test_real)[:,1]
	np.save( "preds" + str(p) + ".npy", preds_real)
	print preds_real
	print "prediction shape:"
	print preds.real.shape



