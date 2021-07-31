"""
@Description: This file contains all the helper functions for Part 1: Feature Extraction

@ Authors:
    - Daniel Bozinovski
    - James Dang
"""

# Imports
import numpy as np
import cv2
import pydicom
import skimage

"""
@Description: This function will take an image and extract all of the above features
@Input: Dicom image pixel array
@Output: Returns the number of non-zero elements in the pixel array
"""


def getImageArea(image):
    return np.count_nonzero(image)


"""
@Description: This function gets us the perimeter of an image
@Input: Dicom image pixel array of edges of image
@Output: Returns the number of non-zero elements in the perimeter pixel array
"""


def getImagePerimeter(image):
    return np.count_nonzero(image)


"""
@Description: This function gives us the irregularity in a given image
@Input: Perimeter and Area of the image
@Output: Returns the irregularity index
"""


def getImageIrregularity(perimeter, area):
    return ((area*4*np.pi) / (perimeter**2))


"""
@Description: This function gives us the equivalenet diameter from the area
@Input: Takes the area of the given image
@Output: Returns the equivalent image diameter
"""


def getImageEquivalentDiameter(area):
    return (np.sqrt((area*4) / np.pi))


"""
@Description: This function gives us an images hu moments
@Input: Takes in the contours of the given image
@Output: Returns the various hu moments besides the 3rd and 7th
"""


def getImageHuMoments(contour):
    hu = cv2.HuMoments(cv2.moments(contour)).ravel().tolist()  # Get the hu's
    hu.pop(-1)  # Remove last hu
    hu.pop(2)  # Remove third hu
    # Return the log of the hu's
    return ([-np.sign(h)*np.log10(np.abs(h)) for h in hu])


"""
@Description: This function will take an image and extract all of the above features

@Input: An image that has been read with pydicom

@Output: Returns the extract features

@Credit: This extraction function was borrowed from @suryathiru (https://www.kaggle.com/suryathiru/1-tradition-image-processing-feature-extraction/)
"""


def extractFeatures(image):

    mean = image.mean()  # Mean
    stdDev = image.std()  # Standard deviation
    equalized = cv2.equalizeHist(image)  # Hist Equalisation

    # Sharpening
    hpf_kernel = np.full((3, 3), -1)
    hpf_kernel[1, 1] = 9

    sharpened = cv2.filter2D(equalized, -1, hpf_kernel)

    ret, binarized = cv2.threshold(cv2.GaussianBlur(
        sharpened, (7, 7), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # thresholding
    # Edge detection for binarized image
    edges = skimage.filters.sobel(binarized)

    # Moments from contours
    contours, hier = cv2.findContours(
        (edges * 255).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    select_contour = sorted(
        contours, key=lambda x: x.shape[0], reverse=True)[0]

    # Return extracted features
    return (mean, stdDev, getImageArea(binarized), getImagePerimeter(edges),
            getImageIrregularity(ar, per), getImageEquivalentDiameter(ar), getImageHuMoments(select_contour))


"""
@Description: This function goes through the dicom image information and returns 1 or 0
              depending on whether the image contains Pneumonia or not

@Inputs: A dataframe containing the metadata

@Output: Returns our test y
"""


def createY(df):

    y = (df['SeriesDescription'] == 'view: PA')
    Y = np.zeros(len(y))  # Initialise Y

    for i in range(len(y)):
        if(y[i] == True):
            Y[i] = 1

    return Y


"""
@Description: Reads each impage path into a list and returns the list

@Inputs: An array of file paths

@Output: Returns our array of read images
"""


def readDicomData(data):

    res = []

    for filePath in tqdm(data):  # Loop over data
        # Read image and stop before pixels to save memory
        f = pydicom.read_file(filePath, stop_before_pixels=True)
        res.append(f)

    return res


"""
@Description: This function parses the medical images meta-data contained

@Inputs: Takes in the dicom image after it has been read

@Output: Returns the unpacked data and the group elements keywords
"""


def parseMetadata(dcm):

    unpackedData = {}
    groupElemToKeywords = {}

    for d in dcm:  # Iterate here to force conversion from lazy RawDataElement to DataElement
        pass

    # Un-pack Data
    for tag, elem in dcm.items():
        tagGroup = tag.group
        tagElem = tag.elem
        keyword = elem.keyword
        groupElemToKeywords[(tagGroup, tagElem)] = keyword
        value = elem.value
        unpackedData[keyword] = value

    return unpackedData, groupElemToKeywords
