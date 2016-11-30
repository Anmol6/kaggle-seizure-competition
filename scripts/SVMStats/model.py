#SVM for ffts
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier as BC 
from sklearn.ensemble import RandomForestClassifier 
import pickle
#folder cheng was using 'data/ffts/train_' + str(patient)+ '_npy/X_new.npy'
# fodler anmol is using '/media/anmol/My Passport/kaggle/ffts

def load_data(patient):
	folder = 'data/stats/'
        X_all = np.load(folder + 'train_' + str(patient)+ '_npy/X_train.npy')
	y_all = np.load(folder + 'train_' + str(patient)+ '_npy/y_train.npy') 
	X_sub = np.load(folder + 'test_' + str(patient)+ '_new/X_test.npy') 
        kpca = pickle.load(open(folder + 'train_' + str(patient)+ '_npy/kpca.p','rb'))
        return X_all, y_all, X_sub, kpca 

patients  = [1,2,3]

linsvc_params = {'penalty':'l2',
              'loss':'squared_hinge', 
              'dual':False,
              'C':1.0, 
              'intercept_scaling':1e4, 
              'class_weight':'balanced',
              #'probability':True,
              #'verbose':1,
              'random_state':42}


svc_params = {'kernel':'linear',
              'C':10.0, 
              'probability':True,
              'class_weight':'balanced',
              #'probability':True,
              #'verbose':1,
              'random_state':42}

bc_params = {'base_estimator':SVC(**svc_params),#LinearSVC(**svc_params),
             'n_estimators':364, 
             'max_samples':0.15, 
             'max_features':0.75,  
             # if you have tons of memory (i.e. 32gb ram + 32gb swap)
             #  incresaing this parameter may help performance.  else,
             #  increasing it may cause "out of memory" errors.
             'n_jobs':6,
             #'n_jobs':8,
             'verbose':1,
             'random_state':42}

submission_preds = []

for p in patients:
	X_all, y_all, X_sub, kpca = load_data(p)
	X_all = np.swapaxes(X_all, 1, 2)

	X_all = X_all.reshape(X_all.shape[0],X_all.shape[1]*X_all.shape[2]*X_all.shape[3])
	print(X_all.shape)
	X_sub = X_sub.reshape(X_sub.shape[0],X_sub.shape[1]*X_sub.shape[2]*X_sub.shape[3])
       
       
        X_all = kpca.transform(X_all) 
        X_sub = kpca.transform(X_sub) 
	print(X_all.shape)
	
        # one hot encode
	y_s = np.zeros((y_all.shape[0], 2))
	y_s[:, 1] = (y_all == 1).reshape(y_all.shape[0],)
	y_s[:, 0] = (y_all == 0).reshape(y_all.shape[0],)
        pos_weight = np.sum(y_s[:,0])/np.sum(y_s[:,1])
	
        X_train,X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.30, stratify=y_all, random_state = 42) 
	del X_all
        
        clf = BC(**bc_params)
        clf.fit(X_train,y_train.ravel())
        
        #clf = RandomForestClassifier(min_samples_leaf=15, class_weight = 'balanced')#{0:1,1:pos_weight})
        #clf.fit(X_train, y_train.ravel())
        
        #print ('oob_score: ', clf.oob_score_)
        # svm = SVC(probability = True, class_weight = 'balanced')
	# svm.fit(X_train, y_train)
	
        preds_tr = clf.predict_proba(X_train)
	print("Predictions_tr:")
        print(preds_tr[0:20])
        print(np.median(preds_tr[:,1]))
	roc_auc_tr = metrics.roc_auc_score(y_train.ravel(), preds_tr[:,1])
	print('ROC AUC_tr:', roc_auc_tr)
        
        preds = clf.predict_proba(X_test)
	print("Predictions:")
        print(preds[0:20])
        print(np.median(preds[:,1]))
	print("Cross_val score:")
	print(clf.score(X_test,y_test.ravel()))
	roc_auc = metrics.roc_auc_score(y_test.ravel(), preds[:,1])
	print('ROC AUC:', roc_auc)
        
        # evaluate and save submission
        submission_preds.append(clf.predict_proba(X_sub)[:,1])

sample_submission = pd.read_csv('submissions/sample_submission.csv')
print(len(submission_preds))
preds_submission = np.concatenate(tuple(submission_preds))
sample_submission['Class'] = preds_submission
sample_submission.to_csv('submissions/SVMKPCAStats007.csv', index=False)
