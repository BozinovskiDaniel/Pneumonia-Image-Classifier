"""
@Description: This file goes through extracting the features for our standard models 

@ Authors:
    - Daniel Bozinovski
    - James Dang
"""

# Imports
import os
import glob
import cv2
import pydicom
import skimage
import numpy as np
import pandas as pd
from tqdm import tqdm
from os import listdir
from tqdm import tqdm_notebook
from os.path import isfile, join
from skimage import feature, filters

import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# File that contains all helper functions
from P1_Helpers import readDicomData, parseMetadata, createY, extractFeatures


# Load in filepaths ("Fix file paths for your ")
trainImagesPath = "./data/rsna-pneumonia-detection-challenge/stage_2_train_images"
testImagesPath = "./data/rsna-pneumonia-detection-challenge/stage_2_test_images"
labelsPath = "./data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
classInfoPath = "./data/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"

labels = pd.read_csv(labelsPath)  # Read Labels
details = pd.read_csv(classInfoPath)  # Read classInfo


fileNames = [f for f in listdir(testImagesPath) if isfile(
    join(testImagesPath, f))]  # Get test image filenames

# Get test data file paths
testFilepaths = glob.glob(f"{testImagesPath}/*.dcm")
testImages = readDicomData(testFilepaths)  # Read test file paths

testMetaDicts, testKeyword = zip(*[parseMetadata(x) for x in tqdm(testImages)])
test_df = pd.DataFrame.from_dict(data=testMetaDicts)  # Convert to dataframe

test_df['dataset'] = 'test'  # Call it test
test_Y = createY(test_df)  # Call the create Y function to get our test Y


featuresTest = []  # Get testing images

for fN in tqdm(fileNames):  # Loop over file names

    path = f"{testImagesPath}/{fN}"  # Get Path

    # Read file and get pixel array
    image = pydicom.read_file(path).pixel_array

    # Extract features & append to array
    featuresTest.append(extractFeatures(image))


featuresTrain = []  # Get training images

# Loop over patient IDs
for patientId in tqdm(labels['patientId']):

    path = f"{trainImagesPath}/{patientId}.dcm"  # Get path

    # Read file and get pixel array
    image = pydicom.read_file(path).pixel_array

    # Extract features & append to array
    featuresTrain.append(extractFeatures(image))

testDf = pd.DataFrame(test_Y, columns=['Target'])

testDf['features'] = featuresTest
labels['features'] = featuresTrain  # Set features

df = pd.DataFrame(labels)

testDf.to_csv('testImageFeatures.csv')  # Download test data as csv
df.to_csv('dicomImageFeatures.csv')  # Download train data as csv
