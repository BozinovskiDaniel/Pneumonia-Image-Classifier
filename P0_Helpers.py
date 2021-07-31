
"""
@Description: This file contains all helper functions for the Part 0 EDA

@ Authors:
    - Daniel Bozinovski
    - James Dang
"""

# Imports
import numpy as np
import pydicom
import pylab as pl

imgsPath = "/Users/Bozinovski/Desktop/UNSW/21t2/COMP9417/Project/Data/rsna-pneumonia-detection-challenge/stage_2_train_images/"

"""
Credit for @peterchang77 for these 2 functions
"""

# This function will allow us to overlay a box


def overlayBox(im, box, rgb, stroke=1):

    # --- Convert coordinates to integers
    box = [int(b) for b in box]

    # --- Extract coordinates
    x, y, width, height = box
    y2 = y + height
    x2 = x + width

    im[y:y + stroke, x:x2] = rgb
    im[y2:y2 + stroke, x:x2] = rgb
    im[y:y2, x:x + stroke] = rgb
    im[y:y2, x2:x2 + stroke] = rgb

    return im


def drawBox(data):

    d = pydicom.read_file(data['dicom'])  # Open and read the file
    im = d.pixel_array

    # Convert to 3 RGB
    im = np.stack([im] * 3, axis=2)

    # Add the boxes with random colours
    for box in data['boxes']:

        rgb = np.floor(np.random.rand(3) * 256).astype('int')  # Get rgb

        im = overlayBox(im=im, box=box, rgb=rgb, stroke=6)  # Overlay the box

    pl.imshow(im, cmap=pl.cm.gist_gray)  # Show the image
    pl.axis('off')  # Remove axis

# We want to create a Data parser to group a patients boxes with its image


def parseData(df):

    newData = {}

    for n, row in df.iterrows():

        patientID = row['patientId']  # Initialise patient

        # If patient is not in the dict, add them
        if patientID not in newData:
            newData[patientID] = {
                'dicom': f"{imgsPath}{patientID}.dcm",
                'classifier': row['Target'],
                'boxes': []}

        # Add box if the patient has pneumonia
        if newData[patientID]['classifier'] == 1:
            newData[patientID]['boxes'].append(
                [row['x'], row['y'], row['width'], row['height']])

    return newData
