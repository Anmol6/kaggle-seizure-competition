### Solution to Kaggle competition: Melbourne University Seizure prediction

Link: https://www.kaggle.com/c/melbourne-university-seizure-prediction

A given dataset consisted of samples of 10-minute long EEG observations from people diagnozed with epillepsy. The challenge is to predict whether a given 10-minute observation is a pre-ictal, meaning it's 1-hour before the seizure, or inter-ictal, normal state when no seizures occur.
  
There were significant challenges faced due to the specifics of the domain of the given problem.  
1) There are far more inter-ictal observations than pre-ictal due to the rare occurences of seizures. This was tackled by scaling the loss of pre-ictal observations in the training phase.  
2) The dataset is very big and doesn't fit on our computers' RAM. Fortunately, raw data is not necessary for accurate forecasting. FFT's were computed that significantly reduced the size of the dataset by several orders of magnitude.

## Pre-processing:
1) Raw Spectrogram
![alt tag](https://github.com/Anmol6/kaggle-seizure-competition/blob/master/img/postprocessfft12band_1.png)

2) Bag of Features

## Techniques that have been tried:

Deep Models on raw spectrograms

# 1) Bi-directional LSTM

Initially we attempted to train LSTM models on raw time series signal.
However due to the large size of the data training times were prohibitively long especially for hyper paramater tuning.
We also tried strided time series with hamming windows.
In general we found that training from raw time series signals led to strong overfitting given the limited dataset.

Much better results were achieved by using spectrograms.
We tried spectrograms with various resolutions in time and frequency.
Eventually we settled on an ensemble of long 1200 timestep and short 300 timestep spectrograms with 12 frequency bands each.
We also found that including standard deviation for each timestep also improved accuracy. From limited testing, it seemed that other statistical features did not have a large impact on accuracy, although time did not permit further investigation.
Here we used a 1D Convolutional layers to feed into a Bi-directional LSTM RNN.
We tried various settings for the convolutional layers. We tried kernels of size 1, 3 and 5 in the time direction.
5 performed poorly, due to overfitting, but 3 performed slightly better than 1 being able to capture some of the temporal difference features of the spectrogam.

# 2) Convolutional Neural Network

The convolutional neural network trained on highly compressed spectrogram with only 20 steps and 6 bands. 1D Convolutions in time were used with no stride and kernel size 1.
A global pooling layer was implemented and used to capture statistical properties between time steps.
The output was fed through a densely connected layer.

Statistical Models on bag of features

# 3) XGBoost

The king of kaggle competitions performed admirably and was the fastest model to iterate on due to its speed and robustness to hyperparameterization. Additionally, it suffered very little from feature selection and so we included almost all computed features in training XGBoost.

# 4) Random Forest

Similar to XGBoost, random forest required minimal adjustment and performed well on bagged features.

# 5) SVM

Bag of features were fed through an RBF kernel PCA to reduce correlated dimensions, imrpove regularization and introduce non-linearity to the model. Bagged Linear SVMs were trained on the kPCA outputs to produce decent probability distributions. Seperate SVMs were also trained with different subsets of the features in order to get a smoother decision boundary and better regularization. This approach worked better than using softmax SVMs.

# 6) Logistic

Logistic Regression was briefly attempted initially, but was unable to produce appreciable accuracy.

## Post-processing:

## Improvements and winning solutions:


## References:
