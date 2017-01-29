# Solution to Kaggle competition: Melbourne University Seizure prediction

Link: https://www.kaggle.com/c/melbourne-university-seizure-prediction

A given dataset consisted of samples of 10-minute long EEG observations from people diagnozed with epillepsy. The challenge is to predict whether a given 10-minute observation is a pre-ictal, meaning it's 1-hour before the seizure, or inter-ictal, normal state when no seizures occur.
  
There were significant challenges faced due to the specifics of the domain of the given problem.  
1) There are far more inter-ictal observations than pre-ictal due to the rare occurences of seizures. This was tackled by scaling the loss of pre-ictal observations in the training phase. Other methods like SMOTE and data augmentation were considered but scaled loss turned out to be the easiest.
2) The dataset is very big and doesn't fit on our computers' RAM. Fortunately, raw data is not necessary for accurate forecasting. FFT's were computed that significantly reduced the size of the dataset by several orders of magnitude yet containing the necessary features to be learnt.

## Pre-processing:
1) Raw Spectrogram
![alt tag](https://github.com/Anmol6/kaggle-seizure-competition/blob/master/img/postprocessfft12band_1.png)

2) Bag of Features
Before the computation of FFTs, the raw signal was filtered from large deviations of very negative and very large voltages. A large collection of features were computed. Most of these features were computed on ~30s time windows with no overlap although some were computed across the whole sample. Some models also used 60 second time windows.
The majority of these features were selected based on related literature and some were general statistical features for time series signals.
Then the data was scaled across time. This was done instead of scaling across features as the time windows have inherent independence in our models. 



## Techniques that have been tried:

Deep Models on raw spectrograms

### 1) Bi-directional LSTM

Initially we attempted to train LSTM models on raw time series signal.
However due to the large size of the data training times were prohibitively long especially for hyper paramater tuning.
We also tried strided time series with hamming windows.
In general we found that training from raw time series signals led to strong overfitting given the limited dataset.

Much better results were achieved by using spectrograms.
We tried spectrograms with various resolutions in time and frequency.
Eventually we settled on an ensemble of long 1200 timestep (with 50% overlap) and short 300 timestep (with no overlap) spectrograms with 12 frequency bands each.
We also found that including standard deviation for each timestep also improved accuracy. From limited testing, it seemed that other statistical features did not have a large impact on accuracy, although time did not permit further investigation.
Here we used a 1D Convolutional layers to feed into a Bi-directional LSTM RNN.
We tried various settings for the convolutional layers. We tried kernels of size 1, 3 and 5 in the time direction.
5 performed poorly, due to overfitting, but 3 performed slightly better than 1 being able to capture some of the temporal difference features of the spectrogam.

### 2) Convolutional Neural Network
Eventually, the main input design matrix looked like 3-D volume of shape frequency x time-windows x channels. This enforced an idea of using a convolution to capture dependencies as CNNs can capture invariant relationships in volumes of data as done in computer vision.

The convolutional neural network trained on highly compressed spectrogram with only 20 steps and 6 bands. 1D Convolutions in time were used with stride of 1 and kernel length of 1, thus making no overlap. 1D convolutions allowed us to capture the invariancy in frequency along just 1 time window of 60 seconds as previous findings showed that it's enough to produce accurate predictions in EEGs. 
A global pooling layer was implemented and used to capture statistical properties between time steps.
Several convolutional layers were implemented in the beginning of the model with relu, second last layer was fully connected with relu too and output was computed with softmax. 

A model in keras has a wrapper with scikit-learn classifier that allowed us to use RandomSearch to find optimal hyperparameters with a stratified k-fold cross-validation. However, due to the inflexibility of this dependency several features that we tried to execute (like Early Stopping) didn't not work in this setting. Given additional computational resources we could have run a larger random search with repititions and with larger number of folds. However, due to the time and compute constraints, the full power of convolution was not a big contribution to the achieved accuracy.


### 3) XGBoost

The king of kaggle competitions performed admirably and was the fastest model to iterate on due to its speed and robustness to hyperparameterization. Additionally, it suffered very little from feature selection and so we included almost all computed features in training XGBoost.

### 4) Random Forest

Similar to XGBoost, random forest required minimal adjustment and performed well on bagged features.

### 5) SVM

Bag of features were fed through an RBF kernel PCA to reduce correlated dimensions, imrpove regularization and introduce non-linearity to the model. Bagged Linear SVMs were trained on the kPCA outputs to produce decent probability distributions. Seperate SVMs were also trained with different subsets of the features in order to get a smoother decision boundary and better regularization. This approach worked better than using softmax SVMs.

### 6) Logistic

Logistic Regression was briefly attempted initially, but was unable to produce appreciable accuracy.

## Post-processing:

We tried a few methods for ensembling the models. One significant feature of the competition was that we were evaluated on combined AUC over all patients rther than the average so median centering the distributions between patients helped to reduce variance due to the False/Positive distribution in the training set.
For combining solutions we simply computed the geometric mean of the various models, scaled loosely based on their performance and generilization ability. The geometric mean seemed to performed better than the simple arithmetic mean and slightly better than the harmonic mean.
We briefly investigated other methods of ensembling such as boosting and bayesian ensembling but lacked the time to evaluate these fully.


## References:
Prediction of seizure likelihood with a long-term, implanted seizure advisory system in patients with drug-resistant epilepsy: a first-in-man study. http://www.sciencedirect.com/science/article/pii/S1474442213700759
Machine Learning for Seizure Prediction:A Revamped 
Approach. http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7275767&tag=1
Predicting epileptic seizures. http://irakorshunova.github.io/2014/11/27/seizures.html
