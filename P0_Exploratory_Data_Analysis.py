"""
@Description: This file goes through some exploratory data analysis

@ Authors:
    - Daniel Bozinovski
    - James Dang
"""

# Imports
import cv2
import tqdm
import pydicom
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import skimage
from skimage import feature, filters

import os
from os import listdir
from os.path import isfile, join

# Get Helper functions
from P0_Helpers import overlayBox, drawBox, parseData

# Get path labels and import the csv files
pathLabels = "/Users/Bozinovski/Desktop/UNSW/21t2/COMP9417/Project/Data/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"
classInfo = "/Users/Bozinovski/Desktop/UNSW/21t2/COMP9417/Project/Data/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"

labels = pd.read_csv(pathLabels)
classInfo = pd.read_csv(classInfo)

# Merge class info and labels
merged = pd.merge(left=classInfo, right=labels,
                  how='left', on='patientId')  # Merge
merged = merged.drop_duplicates()  # Remove duplicates

# How many unique features?
print(f"Unique features: \n{merged.nunique()}")

neg, pos = merged["Target"].value_counts()

print(
    f'How many people have pneumonia vs do not: \n{merged["Target"].value_counts()}\n')

print(
    f'This is roughly {round(pos/(pos+neg)*100, 2)}% having pneumonia within this training data')

# See how many people with pneumonia vs non-pneumonia
merged['Target'].hist()


# Look at the amounts in each of the different classes
merged['class'].hist()

# ===== Part 1: Some Numerical Analysis =====

# Make a copy of just the boxes
boxNums = merged.dropna()[['x', 'y', 'width', 'height']].copy()

# Calculate x2 & y2 coordinates
boxNums['x2'] = boxNums['x'] + boxNums['width']
boxNums['y2'] = boxNums['y'] + boxNums['height']

# Calculate x2 & y2 centres
boxNums['xCentre'] = boxNums['x'] + boxNums['width']/2
boxNums['yCentre'] = boxNums['y'] + boxNums['height']/2

# Calculate area of the box
boxNums['boxArea'] = boxNums['width'] * boxNums['height']

# Look at the correlations between x, y, x2, y2, width, height and the centres

pairs = [(boxNums['x'], boxNums['y']), (boxNums['x2'], boxNums['y2']), (boxNums['width'], boxNums['height']),
         (boxNums['xCentre'], boxNums['yCentre'])]

axis = [(0, 0), (0, 1), (1, 0), (1, 1)]

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

for i in range(4):
    axs[axis[i][0], axis[i][1]].hist2d(pairs[i][0], pairs[i][1], bins=30)

# Set titles
axs[0, 0].set_title('X vs Y')
axs[0, 1].set_title('X2 vs Y2')
axs[1, 0].set_title('Width vs Height')
axs[1, 1].set_title('X Centre vs Y Centre')

plt.show()


# Take a look at the distribution of box area
boxNums['boxArea'].plot(kind='hist',
                        figsize=(14, 4),
                        bins=25,
                        title='Area Distribution of boxes for a Positive target')


# ===== Part 3: Take a Look at the Dicom Images =====

# Get two patients (one who has pneumonia & one who doesnt)
patient0 = merged['patientId'][0]  # Doesn't have pneumonia
patient1 = merged['patientId'][4]  # Has pneumonia

patients = [(patient0, "Doesn't Have Pneumonia"), (patient1, "Has Pneumonia")]

# Plot the images side by side for visual comparison
imgsPath = "/Users/Bozinovski/Desktop/UNSW/21t2/COMP9417/Project/Data/rsna-pneumonia-detection-challenge/stage_2_train_images/"
fig, ax = plt.subplots(1, 2, figsize=(7, 7))

for i in range(2):

    patientID, title = patients[i][0], patients[i][1]  # Extract patient data

    dcmFile = f"{imgsPath}{patientID}.dcm"  # Get path
    dcmData = pydicom.read_file(dcmFile)  # Read file

    img = dcmData.pixel_array  # Get the pixel array

    ax[i].imshow(img, cmap=pl.cm.gist_gray)  # Plot
    ax[i].set_title(title)  # Set title
    ax[i].axis('off')  # Remove axis


parsedData = parseData(merged)

# Check patient 1 which we know has pneumonia
print(parsedData[patient1])

drawBox(parsedData[patient1])  # Draw the box for patient 1


# Get all patients with no pneumonia
patients0 = [(row['patientId'])
             for n, row in merged.iterrows() if row['Target'] == 0]


fig = plt.figure(figsize=(20, 10))

columns = 6
rows = 4

for i in range(1, columns*rows + 1):

    fig.add_subplot(rows, columns, i)  # Add the subplot
    drawBox(parsedData[patients0[i]])  # Draw the box

# Get all patients with pneumonia
patients1 = [(row['patientId'])
             for n, row in merged.iterrows() if row['Target'] == 1]


fig = plt.figure(figsize=(20, 10))

columns = 6
rows = 4

for i in range(1, columns*rows + 1):

    fig.add_subplot(rows, columns, i)  # Add the subplot
    drawBox(parsedData[patients1[i]])  # Draw the box
