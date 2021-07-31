"""
@Description: This file goes through our hyper-parameter tuning / Cross Validation for our Deep Learning Models

@ Authors:
    - Daniel Bozinovski
    - James Dang
"""

# Imports
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

from functools import partial
from collections import defaultdict
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from tqdm import tqdm

# Tensorflow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16
from keras import models
from keras import layers

# sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix

# Helpers
from P3_Helpers import readDicomData, parseMetadata, createY, decodeImage, plottingScores, performCV, build_fcnn_model, build_cnn_model, build_mn_model


# List our paths
trainImagesPath = "../input/rsna-pneumonia-detection-challenge/stage_2_train_images"
testImagesPath = "../input/rsna-pneumonia-detection-challenge/stage_2_test_images"

labelsPath = "../input/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
classInfoPath = "../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"

# Read the labels and classinfo
labels = pd.read_csv(labelsPath)
details = pd.read_csv(classInfoPath)

# Get an array of the test & training file paths
trainFilepaths = glob.glob(f"{trainImagesPath}/*.dcm")
testFilepaths = glob.glob(f"{testImagesPath}/*.dcm")

# Read data into an array
trainImages = readDicomData(trainFilepaths[:13000])
testImages = readDicomData(testFilepaths)


# ===== Balancing Our Data =====

# Number of patients with no pneumonia
COUNT_NORMAL = len(labels.loc[labels['Target'] == 0])
# Number of patients with pneumonia
COUNT_PNE = len(labels.loc[labels['Target'] == 1])
TRAIN_IMG_COUNT = len(trainFilepaths)  # Total patients

# We calculate the weight of each
weight_for_0 = (1 / COUNT_NORMAL)*(TRAIN_IMG_COUNT)/2.0
weight_for_1 = (1 / COUNT_PNE)*(TRAIN_IMG_COUNT)/2.0

classWeight = {0: weight_for_0,
               1: weight_for_1}


# ===== Get Train Y & Test Y =====

# These parse the metadata into dictionaries
trainMetaDicts, trainKeyword = zip(
    *[parseMetadata(x) for x in tqdm(trainImages)])
testMetaDicts, testKeyword = zip(*[parseMetadata(x) for x in tqdm(testImages)])

train_df = pd.DataFrame.from_dict(data=trainMetaDicts)
test_df = pd.DataFrame.from_dict(data=testMetaDicts)

train_df['dataset'] = 'train'
test_df['dataset'] = 'test'

df = train_df
df2 = test_df

train_Y = createY(df)  # Create training Y
test_Y = createY(df2)  # Create testing Y


# ===== Get Train X & Test X =====

# Get our train x in the correct shape
train_X = []

for filePath in tqdm(trainFilepaths[:13000]):

    img = decodeImage(filePath)
    train_X.append(img)

train_X = np.array(train_X)  # Convert to np.array
# Reshape into rgb format
train_X_rgb = np.repeat(train_X[..., np.newaxis], 3, -1)

# Get our test x in the correct shape for NN
test_X = []

for filePath in tqdm(testFilepaths):
    img_test = decodeImage(filePath)  # Decode & Resize
    test_X.append(img_test)

test_X = np.array(test_X)  # Convert to np array
# Reshape into rgb format
test_X_rgb = np.repeat(test_X[..., np.newaxis], 3, -1)


# ===== Metrics =====

# These our our scoring metrics that are going to be used to evaluate our models
METRICS = ['accuracy',
           tf.keras.metrics.Precision(name='precision'),
           tf.keras.metrics.Recall(name='recall'),
           tf.keras.metrics.AUC(name='AUC')]

# ===== Defining Our Callbacks =====

# Define our callback functions to pass when fitting our NNs
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "xray_model.h5", save_best_only=True)

early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=5, restore_best_weights=True)

# Exponential decay function


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return lr0 * 0.1 ** (epoch / s)
    return exponential_decay_fn


exponential_decay_fn = exponential_decay(0.01, 20)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(exponential_decay_fn)


# ===== Building Model #1 - Fully Connected Model =====

# Build our FCNN model and compile
model_fcnn = build_fcnn_model()
model_fcnn.summary()
model_fcnn.compile(optimizer="adam", loss="binary_crossentropy",
                   metrics=METRICS)  # Compile

history_fcnn = model_fcnn.fit(train_X_rgb,
                              train_Y,
                              epochs=30,
                              batch_size=128,
                              validation_split=0.2,
                              class_weight=classWeight,
                              verbose=1,
                              callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler])  # Fit the model

# Evaluate and display results
# Evaluate the model on test data
results = model_fcnn.evaluate(test_X_rgb, test_Y)
results = dict(zip(model_fcnn.metrics_names, results))

print(results)
plottingScores(history_fcnn)  # Visualise scores


# ===== Building Model #2 - Convolutional Neural Network =====

# Build and compile model
model_cnn = build_cnn_model()
model_cnn.summary()
model_cnn.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=METRICS)

# Fit model
history_cnn = model_cnn.fit(train_X_rgb,
                            train_Y,
                            epochs=30,
                            validation_split=0.15,
                            batch_size=128,
                            class_weight=classWeight,
                            callbacks=[checkpoint_cb,
                                       early_stopping_cb, lr_scheduler],
                            verbose=1)  # Fit the model

# Evalute the models results and put into a dict
results = model_cnn.evaluate(test_X_rgb, test_Y)
results = dict(zip(model_cnn.metrics_names, results))

print(results)
plottingScores(history_cnn)  # Visualise scores


# ===== Building Model #3 - MobileNet w/Transfer Learning =====

# Build and compile mobile net model
model_mn = build_mn_model()
model_mn.summary()
model_mn.compile(optimizer='adam', loss='binary_crossentropy', metrics=METRICS)

history_mn = model_mn.fit(train_X_rgb,
                          train_Y,
                          epochs=30,
                          validation_split=0.20,
                          class_weight=classWeight,
                          batch_size=64,
                          callbacks=[checkpoint_cb, early_stopping_cb, lr_scheduler])

# Show results and print graphs
results = model_mn.evaluate(test_X_rgb, test_Y)
results = dict(zip(model_mn.metrics_names, results))

print(results)
plottingScores(history_mn)  # Visualise scores


# ===== Display Confusion Matrix =====

y_pred = model_mn.predict_classes(test_X_rgb)
confusion_matrix(test_Y, y_pred)


# ===== Perform K-Fold Cross Validation

cbs = [checkpoint_cb, early_stopping_cb,
       lr_scheduler]  # Get callbacks in array

# Full-connected NN
resFCNN = performCV(5, build_fcnn_model, 30, 128,
                    train_X_rgb, train_Y, classWeight, cbs)
print(resFCNN)

# Convolutional NN
resCNN = performCV(5, build_cnn_model, 30, 64,
                   train_X_rgb, train_Y, classWeight, cbs)
print(resCNN)

# MobileNet
resMB = performCV(5, build_mn_model, 30, 64,
                  train_X_rgb, train_Y, classWeight, cbs)
print(resMB)
