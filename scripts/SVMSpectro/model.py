#SVM for ffts
import numpy as np
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
 

#folder cheng was using 'data/ffts/train_' + str(patient)+ '_npy/X_new.npy'
# fodler anmol is using '/media/anmol/My Passport/kaggle/ffts

def load_data(patient):
	X_o = np.load('/media/anmol/My Passport/kaggle/ffts_10/train_' + str(patient)+ '_npy/X_new.npy')
	y_o = np.load('/media/anmol/My Passport/kaggle/ffts_10/train_' + str(patient)+ '_npy/y_new.npy') 
	X_n = np.load('/media/anmol/My Passport/kaggle/ffts_10/test_' + str(patient)+ '_npy/X_new.npy')
	y_n = np.load('/media/anmol/My Passport/kaggle/ffts_10/test_' + str(patient)+ '_npy/y_new.npy')

	X_all = np.concatenate((X_o, X_n), axis = 0)
	y_all = np.concatenate((y_o, y_n), axis = 0)
	X_all[X_all == -np.inf] = -10
	X_all[X_all > 400] = 400   
	return X_all, y_all  

patients  = [1,2,3]

for p in patients:
	X_all, y_all = load_data(p)
	X_all = np.swapaxes(X_all, 1, 2)

	X_all = X_all.reshape(X_all.shape[0],X_all.shape[1],X_all.shape[2]*X_all.shape[3])
	print(X_all.shape)

	# one hot encode
	y_s = np.zeros((y_all.shape[0], 2))
	y_s[:, 1] = (y_all == 1).reshape(y_all.shape[0],)
	y_s[:, 0] = (y_all == 0).reshape(y_all.shape[0],)

	X_train,X_test, y_train, y_test = train_test_split(X_all, y_s, test_size=0.25, stratify=y_all) 
	del X_all

	svm = SVC(probability = True, class_weight = 'balanced')
	svm.fit(X_train, y_train)
	preds = svm.predict_proba(y_test)
	print("Predictions:")
	print(preds)
	print("Cross_val score:")
	print(svm.score(X_test,y_test))
	roc_auc = metrics.roc_auc_score(y_test[:,1], preds[:,1])
	print('ROC AUC:', roc_auc)

