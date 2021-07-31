"""
@Description: This file goes our Cross Validation / Hyper-paramter Tuning for Our Standard Models

@ Authors:
    - Daniel Bozinovski
    - James Dang
"""

# Imports
from tqdm import tqdm
from lightgbm import LGBMClassifier
from joblib import Parallel, delayed
from collections import defaultdict
from functools import partial
import os
import cv2
import glob
import time
import pydicom
import skimage
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import feature, filters

# Sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sklearn.metrics as skm

# Features Path
imageFeaturesPath = "../input/extractedfeatures/dicomImageFeatures.csv"
testImageFeaturesPath = "../input/extractedfeatures/testImageFeatures.csv"
labelsPath = "../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"

# Read csv's
imageFeatures = pd.read_csv(imageFeaturesPath)
testImageFeatures = pd.read_csv(testImageFeaturesPath)
labels = pd.read_csv(labelsPath)

"""
@Description: This function will turn the given image features into a dataframe
@Input: Image features 
@Output: Returns the dataframe with the features reorganised as columns
"""


def getFeaturesDF(imgFeatures):

    features = imgFeatures.features.apply(lambda x: list(eval(x)))

    df = pd.DataFrame(features.values.tolist(),
                      columns=['mean', 'stddev', 'area', 'perimeter', 'irregularity',
                               'equiv_diam', 'hu1', 'hu2', 'hu4', 'hu5', 'hu6'],
                      index=imgFeatures.index)

    df['hasPneumonia'] = labels['Target']

    return df


# ===== Get train and test features dataframes =====

trainData = getFeaturesDF(imageFeatures)
testData = getFeaturesDF(testImageFeatures)

# ===== Get Class Weights =====

COUNT_NORMAL = len(trainData.loc[trainData['hasPneumonia'] == 0])
COUNT_PNE = len(trainData.loc[trainData['hasPneumonia'] == 1])
TRAIN_IMG_COUNT = len(trainData)

weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0
weight_for_1 = (1 / COUNT_PNE)*(TRAIN_IMG_COUNT)/2.0

classWeight = {0: weight_for_0, 1: weight_for_1}

# ===== Normalise Data =====

# For Train Data
trainData.dropna()

# Split data into x and y
x = trainData.drop(columns=['hasPneumonia'])
y = trainData['hasPneumonia']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# For Test data
testData.dropna()

# Split data into x and y
x_test_unseen = testData.drop(columns=['hasPneumonia'])
y_test_unseen = testData['hasPneumonia']

# Scale the features
scaler = StandardScaler()
X_scaled_test_unseen = scaler.fit_transform(x_test_unseen)

# ===== Split into Test & Training Data =====

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    stratify=y,
    shuffle=True,
    test_size=0.3)

# ===== Model 1: Logistic Regression Randomised Grid Search =====

# Performing randomised Grid Search
cVals = list(range(1, 6))
cWeight = [None, 'Balanced']

params = dict(C=cVals, class_weight=cWeight)

logReg = LogisticRegression()
clf = RandomizedSearchCV(logReg, params, random_state=0)

search = clf.fit(X_train, y_train)
print(search.best_params_)  # Print the best hyper-parameters


# ===== Model 2: K-Nearest Neighbour Randomised Grid Search =====

# Performing randomised Grid Search
kValues = list(range(10, 210, 10))
weight_options = ["uniform", "distance"]

params = dict(n_neighbors=kValues, weights=weight_options)

kNN = KNeighborsClassifier()
clf = RandomizedSearchCV(kNN, params, random_state=0)

search = clf.fit(X_train, y_train)
print(search.best_params_)  # Print the best hyper-parameters


# ===== Model 4: Random Forest Randomised Grid Search =====

# Perform Randomised Grid Search
classWeights = [None, 'Balanced']
nEstimatorValues = list(range(300, 800, 100))
maxDepthValues = list(range(6, 10))
minSamplesSplitValues = list(range(2, 5))

params = dict(n_estimators=nEstimatorValues,
              max_depth=maxDepthValues,
              class_weight=classWeights,
              min_samples_split=minSamplesSplitValues)

rfc = RandomForestClassifier(n_jobs=-1)

clf = RandomizedSearchCV(rfc, params, random_state=0)

search = clf.fit(X_train, y_train)
print(search.best_params_)  # Display best hyper-parameters


# ===== Model 5: Random Forest Randomised Grid Search =====

# Perform Randomised Grid Seach
cVals = np.arange(0.5, 1.6, 0.1)
classWeights = [None, 'Balanced']

params = dict(C=cVals, class_weight=classWeights)

svm = SVC()

clf = RandomizedSearchCV(svm, params, random_state=0)
search = clf.fit(X_train, y_train)
print(search.best_params_)  # Return the best hyper-parametrs


# ===== Model 6: Gradient Boosting Classifier Randomised Grid Search =====

# Perform Randomised Grid Search to Find Optimal Hyper-parameters
nEstimatorValues = list(range(300, 800, 100))
maxDepthValues = list(range(6, 10))
minSamplesSplitValues = list(range(2, 5))
lrs = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]

params = dict(n_estimators=nEstimatorValues,
              max_depth=maxDepthValues,
              learning_rate=lrs,
              min_samples_split=minSamplesSplitValues)


gbc = GradientBoostingClassifier()

clf = RandomizedSearchCV(gbc, params, random_state=0)
search = clf.fit(X_train, y_train)
print(search.best_params_)


# ===== Function to Perform K-Fold Cross Validation
def performCV(model, name, K):

    print(f"===== Performing CV for {name} =====")
    kfold = KFold(n_splits=K, shuffle=True)

    accuracy_per_fold = []
    precision_per_fold = []
    recall_per_fold = []
    mse_per_fold = []
    auc_per_fold = []

    for train_index, test_index in kfold.split(X_scaled):

        # Split data
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)  # Fit data
        pred_y_test = model.predict(X_test)  # Make a prediction

        accuracy = skm.accuracy_score(y_test, pred_y_test)
        precision = skm.precision_score(y_test, pred_y_test)
        recall = skm.recall_score(y_test, pred_y_test)
        mse = skm.mean_squared_error(y_test, pred_y_test)
        auc = skm.roc_auc_score(y_test, pred_y_test)

        accuracy_per_fold.append(accuracy)
        precision_per_fold.append(precision)
        recall_per_fold.append(recall)
        mse_per_fold.append(mse)
        auc_per_fold.append(auc)

    return {
        'mean_accuracy': np.mean(accuracy_per_fold),
        'mean_precision': np.mean(precision_per_fold),
        'mean_recall': np.mean(recall_per_fold),
        'mean_mse': np.mean(mse_per_fold),
        'mean_auc': np.mean(auc_per_fold)
    }


# ===== Perform CV with optimal hyper-paramters from each model

logReg = LogisticRegression(C=1)
kNN = KNeighborsClassifier(150, weights="distance")
gnb = GaussianNB()
rfc = RandomForestClassifier(
    n_estimators=700, max_depth=9, min_samples_split=4, n_jobs=-1)
svm = SVC(C=1.5)
gbc = GradientBoostingClassifier(
    n_estimators=700, max_depth=9, min_samples_split=4, learning_rate=0.005)

modelsList = [(logReg, "Logistic Regression"),
              (kNN, "K-Nearest Neighbour"),
              (gnb, "Naive Bayes"),
              (rfc, "Random Forest"),
              (svm, "Support Vector Machine"),
              (gbc, "Gradient Boosting Classifier")]

CVResults = {}

for m in modelsList:
    CVResults[m[1]] = performCV(m[0], m[1], 5)

print(CVResults)  # Display the Cross Validation results for each model
