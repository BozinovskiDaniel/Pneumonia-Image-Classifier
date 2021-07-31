"""
@Description: This file contains all the helper functions for Part 3: Modelling Deep Learning

@ Authors:
    - Daniel Bozinovski
    - James Dang
"""

import cv2
import tqdm
import pydicom
import tensorflow as tf
import matplotlib as plt
import numpy as np
from sklearn.model_selection import KFold

"""
@Description: Reads an array of dicom image paths, and returns an array of the images after they have been read

@Inputs: An array of filepaths for the images

@Output: Returns an array of the images after they have been read
"""


def readDicomData(data):

    res = []

    for filePath in tqdm(data):  # Loop over data

        # We use stop_before_pixels to avoid reading the image (Saves on speed/memory)
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


"""
@Description: This function goes through the dicom image information and returns 1 or 0
              depending on whether the image contains Pneumonia or not

@Inputs: A dataframe containing the metadata

@Output: Returns the Y result (i.e: our train and test y)
"""


def createY(df):
    y = (df['SeriesDescription'] == 'view: PA')
    Y = np.zeros(len(y))  # Initialise Y

    for i in range(len(y)):
        if(y[i] == True):
            Y[i] = 1

    return Y


"""
@Description: This decodes an image by reading the pixel array, resizing it into the correct format and
              normalising the pixels

@Inputs:
    - filePath: This is the filepath of the image that we want to decode

@Output:
    - img: This is the image after it has been decoded
"""


def decodeImage(filePath):
    image = pydicom.read_file(filePath).pixel_array
    image = cv2.resize(image, (128, 128))
    return (image/255)


"""
@Description: This function plots our metrics for our models across epochs

@Inputs: The history of the fitted model

@Output: N/A
"""


def plottingScores(hist):
    fig, ax = plt.subplots(1, 5, figsize=(20, 3))
    ax = ax.ravel()

    for i, met in enumerate(['accuracy', 'precision', 'recall', 'AUC', 'loss']):
        ax[i].plot(hist.history[met])
        ax[i].plot(hist.history['val_' + met])
        ax[i].set_title('Model {}'.format(met))
        ax[i].set_xlabel('epochs')
        ax[i].set_ylabel(met)
        ax[i].legend(['train', 'val'])


"""
@Description: This function builds our simple Fully-connected NN

@Inputs: N/A

@Output: Returns the FCNN Model
"""


def build_fcnn_model():

    # Basic model with a flattening layer followng by 2 dense layers
    # The first dense layer is using relu and the 2nd one is using sigmoid
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(128, 128, 3)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    return model


"""
@Description: This function builds our custom CNN Model

@Inputs: N/A

@Output: Returns the CNN model
"""


def build_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(
            1, 1), padding='valid', activation='relu', input_shape=(128, 128, 3)),  # convolutional layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # flatten output of conv

        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), strides=(
            1, 1), padding='valid', activation='relu'),  # convolutional layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),  # flatten output of conv
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(64, 3, activation='relu', padding='valid'),
        tf.keras.layers.Conv2D(128, 3, activation='relu', padding='valid'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Flatten(),  # flatten output of conv
        tf.keras.layers.Dense(512, activation="relu"),  # hidden layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation="relu"),  # output layer
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation="sigmoid")])

    return model


"""
@Description: This function builds our MobileNet Model

@Inputs: N/A

@Output: Returns the Mobile Net model
"""


def build_mn_model():

    model = tf.keras.Sequential([
        tf.keras.applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=(128, 128, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.layers[0].trainable = False

    return model


"""
@Description: This function performs K-Fold Cross Validation with a provided Deep Learning Model

@Inputs:
    - K: Number of folds
    - build_model_func: Function to create model
    - epochs: Number of epochs to train data
    - batchSize: Batch size when fitting the model

@Output: Dict of metric results from K-fold CV
"""


def performCV(K, build_model_func, epochs, batchSize, train_X_rgb, train_Y, classWeight, cbs):

    kfold = KFold(n_splits=K, shuffle=True)  # Split data into K Folds

    res = {
        'acc_per_fold': [],
        'precision_per_fold': [],
        'recall_per_fold': [],
        'auc_per_fold': [],
        'loss_per_fold': []
    }

    fold_no = 1

    for train_index, test_index in kfold.split(train_X_rgb):

        # Split data
        X_train, X_test = train_X_rgb[train_index], train_X_rgb[test_index]
        y_train, y_test = train_Y[train_index], train_Y[test_index]

        model = build_model_func()  # Build model
        mets = ['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(
            name='recall'), tf.keras.metrics.AUC(name='AUC')]

        model.compile(optimizer='adam', loss='binary_crossentropy',
                      metrics=mets)  # Compile our model

        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Train the model on the current fold
        history = model.fit(X_train,
                            y_train,
                            epochs=epochs,
                            batch_size=batchSize,
                            class_weight=classWeight,
                            callbacks=[cbs[0], cbs[1], cbs[2]])  # Fit data to model

        scores = model.evaluate(X_test, y_test, verbose=0)  # Evalute the model

        print(f'Scores for fold {fold_no}:')
        print(f'{model.metrics_names[0]}: {scores[0]}')
        print(f'{model.metrics_names[1]}: {scores[1]*100}%')
        print(f'{model.metrics_names[2]}: {scores[2]*100}%')
        print(f'{model.metrics_names[3]}: {scores[3]*100}%')

        res['loss_per_fold'].append(scores[0])
        res['acc_per_fold'].append(scores[1] * 100)
        res['precision_per_fold'].append(scores[2]*100)
        res['recall_per_fold'].append(scores[3]*100)
        res['auc_per_fold'].append(scores[4]*100)

        # Increase fold number
        fold_no += 1

    return res  # return our results dict
