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
	folder = 'data/stats1/'
    X_train = np.load(folder + 'train_' + str(patient)+ '_npy/X_fstrain.npy')
	y_train = np.load(folder + 'train_' + str(patient)+ '_npy/y_fstrain.npy') 
    X_test = np.load(folder + 'train_' + str(patient)+ '_npy/X_fsval.npy')
	y_test = np.load(folder + 'train_' + str(patient)+ '_npy/y_fsval.npy') 
	X_sub = np.load(folder + 'test_' + str(patient)+ '_new/X_test.npy') 
	return X_train, y_train, X_test, y_test, X_sub 

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
             'max_samples':0.15, 
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
	X_train, y_train, X_test, y_test, X_sub = load_data(p)
	print(X_train.shape)
    X_train = np.swapaxes(X_train, 1, 2)
    X_test = np.swapaxes(X_test, 1, 2)
    
	X_train = X_train.reshape(X_train.shape[0],X_train.shape[1]*X_train.shape[2]*X_train.shape[3])
	X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2]*X_test.shape[3])
	print(X_train.shape)
	X_sub = X_sub.reshape(X_sub.shape[0],X_sub.shape[1]*X_sub.shape[2]*X_sub.shape[3])

    rng_state = np.random.get_state()
    np.random.shuffle(X_train)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)

    rng_state = np.random.get_state()
    np.random.shuffle(X_test)
    np.random.set_state(rng_state)
    np.random.shuffle(y_test)


	# one hot encode
    y_s = np.zeros((y_train.shape[0], 2))
	y_s[:, 1] = (y_train == 1).reshape(y_train.shape[0],)
	y_s[:, 0] = (y_train == 0).reshape(y_train.shape[0],)
        '''
        y_test = np.zeros((y_test1.shape[0], 2))
        y_test[:, 1] = (y_test1 == 1).reshape(y_test1.shape[0],)
        y_test[:, 0] = (y_test1 == 0).reshape(y_test1.shape[0],)
        '''

    pos_weight = np.sum(y_s[:,0])/np.sum(y_s[:,1])
	
        #X_train,X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.20, stratify=y_all, random_state = 42) 
	#del X_all
        
        #bc = BC(**bc_params)
        #bc.fit(X_train,y_train.ravel())
        
	eta = 0.2
	max_depth = 3
	subsample = 0.5
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
	print('Length test:', len(X_test))

	dtrain = xgb.DMatrix(X_train, y_train)
	dtest = xgb.DMatrix(X_test, y_test)

	watchlist = [(dtrain, 'train'), (dtest, 'eval')]
	gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist,
			early_stopping_rounds=early_stopping_rounds, verbose_eval=True)

	print("Validating...")
	check = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration+1)
	score = roc_auc_score(y_test, check)
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
sample_submission.to_csv('submissions/XGBOOST003.csv', index=False)
