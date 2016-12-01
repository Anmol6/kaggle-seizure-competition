#SVM for ffts
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier as BC 
from sklearn.ensemble import RandomForestClassifier 
import xgboost as xgb
import time
from sklearn.metrics import roc_auc_score

#folder cheng was using 'data/ffts/train_' + str(patient)+ '_npy/X_new.npy'
# fodler anmol is using '/media/anmol/My Passport/kaggle/ffts
random_state =1337

def load_data(patient):
	folder = 'data/stats/'
        X_all = np.load(folder + 'train_' + str(patient)+ '_npy/X_train.npy')
	y_all = np.load(folder + 'train_' + str(patient)+ '_npy/y_train.npy') 
	X_sub = np.load(folder + 'test_' + str(patient)+ '_new/X_test.npy') 
	return X_all, y_all, X_sub 

patients  = [1,2,3]

svc_params = {'penalty':'l2',
              'loss':'squared_hinge', 
              'dual':False,
              'C':33.0, 
              'intercept_scaling':1e4, 
              'class_weight':'balanced',
              #'verbose':1,
              'random_state':42}

bc_params = {'base_estimator':LinearSVC(**svc_params),
             'n_estimators':96, 
             'max_samples':0.10, 
             'max_features':0.8,  
             'oob_score':True,
             # if you have tons of memory (i.e. 32gb ram + 32gb swap)
             #  incresaing this parameter may help performance.  else,
             #  increasing it may cause "out of memory" errors.
             'n_jobs':6,
             #'n_jobs':8,
             'verbose':1,
             'random_state':42}

submission_preds = []

for p in patients:
	X_all, y_all, X_sub = load_data(p)
	X_all = np.swapaxes(X_all, 1, 2)

	X_all = X_all.reshape(X_all.shape[0],X_all.shape[1]*X_all.shape[2]*X_all.shape[3])
	print(X_all.shape)
	X_sub = X_sub.reshape(X_sub.shape[0],X_sub.shape[1]*X_sub.shape[2]*X_sub.shape[3])
        
	# one hot encode
	y_s = np.zeros((y_all.shape[0], 2))
	y_s[:, 1] = (y_all == 1).reshape(y_all.shape[0],)
	y_s[:, 0] = (y_all == 0).reshape(y_all.shape[0],)
        pos_weight = np.sum(y_s[:,0])/np.sum(y_s[:,1])
	
        X_train,X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.20, stratify=y_all, random_state = 42) 
	del X_all
        
        #bc = BC(**bc_params)
        #bc.fit(X_train,y_train.ravel())
        
	eta = 0.2
	max_depth = 4
	subsample = 0.6
	colsample_bytree = 0.7
	start_time = time.time()

	print('XGBoost params. ETA: {}, MAX_DEPTH: {}, SUBSAMPLE: {}, COLSAMPLE_BY_TREE: {}'.format(eta, max_depth, subsample, colsample_bytree))
	params = {
	    "objective": "binary:logistic",
	    "booster" : "gbtree",
	    "eval_metric": "auc",
	    "eta": eta,
	    "tree_method": 'exact',
	    "max_depth": max_depth,
	    "subsample": subsample,
	    "colsample_bytree": colsample_bytree,
	    "silent": 1,
	    "seed": random_state,
	}
	num_boost_round = 1000
	early_stopping_rounds = 50
	test_size = 0.2

	print('Length train:', len(X_train))
	print('Length valid:', len(X_valid))

	dtrain = xgb.DMatrix(X_train, y_train)
	dvalid = xgb.DMatrix(X_valid, y_valid)

	watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
	gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
			early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

	print("Validating...")
	check = gbm.predict(xgb.DMatrix(X_valid), ntree_limit=gbm.best_iteration+1)
	score = roc_auc_score(y_valid, check)
	print('Check error value: {:.6f}'.format(score))

	print("Predict test set...")
	test_prediction = gbm.predict(xgb.DMatrix(X_sub), ntree_limit=gbm.best_iteration+1)
	print(test_prediction.shape)
	print('Training time: {} minutes'.format(round((time.time() - start_time)/60, 2)))       
        #print ('oob_score: ', clf.oob_score_)
        # svm = SVC(probability = True, class_weight = 'balanced')
	# svm.fit(X_train, y_train)

	'''	
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
        '''
        # evaluate and save submission
        submission_preds.append(test_prediction)

sample_submission = pd.read_csv('submissions/sample_submission.csv')
print(len(submission_preds))
preds_submission = np.concatenate(tuple(submission_preds))
sample_submission['Class'] = preds_submission
sample_submission.to_csv('submissions/XGBOOST002.csv', index=False)
