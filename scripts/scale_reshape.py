from data_scaler import scale_across_features, scale_across_time
from sklearn.preprocessing import StandardScaler
import numpy as np


# Accepts X's of shape (examples, channels, time-windows, bins)
# X_trains and X_tests must be list
def scale_reshape(X_trains, X_tests):
    # applying scale_across_time
    for i, X in enumerate(X_trains):
        X_train = X.swapaxes(2, 3)
        X_test = X_tests[i]
        X_test = X_test.swapaxes(2, 3)
        X_trains[i], scalers = scale_across_time(X_train, X_test)
        X_tests[i], additional_scalers = scale_across_time(X_test, scalers=scalers)
    # swaping the axis back and reshaping the features
    for i, X in enumerate(X_trains):
        X_trains[i] = X.swapaxes(2, 3)
        X_trains[i] = X.reshape((X.shape[0], 10, 112))   
        X = X_tests[i]
        X_tests[i] = X.swapaxes(2, 3)
        X_tests[i] = X.reshape((X.shape[0], 10, 112)) 
    return X_trains, X_tests, scalers
