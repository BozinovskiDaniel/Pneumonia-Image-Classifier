# Classifying Pneumonia in Medical Images

This is a README.md for a ML project from Kaggle.

@Kaggle Competition: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

<hr>

<b><i>Each Notebook Name is hyper-linked to the Kaggle Notebook code in the table below</i></b>

## Contents

| Part |                                                           Notebook                                                           | Explanation                                                                                                                                                                                                                                                                  |
| :--: | :--------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  0   |              [Exploratory Data Analysis](https://www.kaggle.com/danielbozinovski/p0-exploratory-data-analysis)               | Exploring and <b>analysing</b> the given medical images dataset                                                                                                                                                                                                                         |
|  1   | [Feature Extraction for Standard Models](https://www.kaggle.com/danielbozinovski/p1-feature-extraction-for-standard-models)  | Code used to <b>extract image features</b> from the medical images and save to a .csv for the standard models (The CSV's are used for Part 2)                                                                                                                                |
|  2   |   [Modelling Pneumonia - Standard Models](https://www.kaggle.com/danielbozinovski/p2-modelling-pneumonia-standard-models)    | This notebook goes through Optimal Hyper-parameter search & K-Fold Cross Validation for Standard Models. Standard models include: <b>Logistic Regression</b>, <b>kNN</b>, <b>Gaussian Naive Bayes</b>, <b>Random Forest</b>, <b>SVM</b> & <b>Gradient Boosted Classifier</b> |
|  3   | [Modelling Pneumonia - Deep Learning Models](https://www.kaggle.com/danielbozinovski/p3-modelling-pneumonia-neural-networks) | This notebook does through Optimal Hyper-paramter search and K-Fold Cross Validation for Deep Learning Models. DL Models include: <b>Fully-Connected NN</b>, <b>Convolutional NN</b>, <b>Mobile-Net w/ Transfer Learning</b>                                                 |

## Raw Links to Notebooks

- <b><i>P0 - Notebook 1:</i></b> https://www.kaggle.com/danielbozinovski/p0-exploratory-data-analysis

- <b><i>P1 - Notebook 2:</i></b> https://www.kaggle.com/danielbozinovski/p1-feature-extraction-for-standard-models

- <b><i>P2 - Notebook 3:</i></b> https://www.kaggle.com/danielbozinovski/p2-modelling-pneumonia-standard-models

- <b><i>P3 - Notebook 4:</i></b> https://www.kaggle.com/danielbozinovski/p3-modelling-pneumonia-neural-networks

## Overview

We are challenged to build an algorithm to detect for Pneumonia in medical images.

## Backstory

Pneumonia accounts for over 15% of all deaths of children under 5 years old internationally. In 2015, 920,000 children under the age of 5 died from the disease. In the United States, pneumonia accounts for over 500,000 visits to emergency departments and over 50,000 deaths in 2015, keeping the ailment on the list of top 10 causes of death in the country.

## Group Members

- [@DanielBozinovski](https://github.com/BozinovskiDaniel)
- [@JamesDang](https://github.com/realblingy)

## Part 1: Implementation

Within this section, we will discuss all the various aspects that we considered when building our models, including:
- Reviewing what algorithms work most favourably with a binary classification problem
-	Looking at the Convolutional Neural Network architectures tested
-	Our process for selecting the most optimal hyper-parameters for our models
-	Our process of extracting features from patient images to build our models upon
-	Our evaluation metrics used
-	How we selected our best model to run against the final test data

### Part 1.1: Reviewing Relevant Algorithms

Literature regarding the best ways to classify an image almost all pointed towards utilising Neural Networks, specifically a special type of neural network called a Convolutional Neural Network (CNN). (MediumDataDrivenInvestor, 2019) This is because CNNs automatically detect the most important features of an image without any human supervision. Also, the concept of dimensionality reduction perfectly suits the significant number of parameters in each image. (MediumDataDrivenInvestor, 2019)

Another important feature of CNNs is that the number of parameters don’t scale by the size of the original image (i.e.: they are independent of the inputted images size). (TowardsDataScience, 2018) Fortunately for us, many deep learning models were implemented in the open-source neural network library Keras. 

Regarding the standard Sklearn models learned throughout the course, we still attempted to use them to get an educational base-line comparison for how much better the Deep Neural Networks performed. To do so, we instead had to perform some traditional image pre-processing methods to extract the features from our images to use in our standard models.

We have included the diagram (AISummer, 2021) to see a comparison with the number of parameters/operations to the accuracy for many various CNN architectures. We can see that it is evident that more parameters do not necessarily lead to better accuracy.

![image](https://user-images.githubusercontent.com/47773746/128126705-28d66f5d-36d3-4640-aca5-289a47306fdd.png)

### Part 1.2: Convolutional Neural Networks Tested

Within this section, we go through the various stages of our CNNs architectures. We progressively built our network, by adding more convolutions (making our network deeper) and adding max pooling to reduce variance and computations throughout. As we make our network deeper, we will notice that our network will begin overfitting, so to attempt to counteract this, we can begin adding drop-out layers which will randomly set the outgoing edges of hidden units to 0 at each update of the training phase. (TowardsDataScience, 2019)

We also want to note that we usually want to place dropouts (ones with a greater drop-out rate) on the fully connected layers (towards the end) because they are the ones with the largest number of parameters, which would make them likely to excessively adapt and cause overfitting. (StackOverflow, 2017)

According to literature, the basic principle that should be followed when building a CNN is to ‘keep the feature space wide and shallow in the initial stages of the network, and then make it narrower and deeper towards the end’. (TowardsDataScience, 2018) Note: Since the borders of the medical images aren’t very important for this classification problem (we are mainly looking at the centre where the lungs are), we didn’t apply any padding to the input layers.


### Part 1.3: Hyper-parameter Tuning

The hyper-parameters that were considered for each learning algorithm are summarised in Table 1. Model Hyperparameters are the process governing the entire training process. Hyper-parameters are crucial in directly controlling the behaviour of the training algorithm, having a huge impact on the performance of the model being trained. For example, if the learning rate for a model is too low, the model may miss the important patterns of the data, but if it is too high it may have collisions. (TowardsDataScience, 2018)

To tune our hyper-parameters, we utilised Sklearn’s Random Grid Search. This was used instead of Grid Search, which is significantly more computationally intensive and would have taken us too long. Random Grid Search instead works by randomly sampling the entire search space and evaluating sets from a specified probability distribution.

Regarding our Deep Learning models, we utilised a large amount of manual hyper-parameter tuning as using tools such as Grid Search would take far too long to train some models. We also employed call-back functions from the Keras library to improve our Neural Networks performance. A call-back is a set of functions to be applied at given stages of the training procedure. Call-backs give you an internal view of the model’s state and statistics during training. The main 3 call-backs that we have used include:
-	Early Stopping – one of the more important call-backs as it actively works to prevent over-fitting when training our models. We know that training over too many epochs can result in a model overfitting to the training dataset, whilst training over too little epochs will result in the model underfitting. Early stopping is essentially a method that allows you to specify many epochs to train over and stops training once the model’s performance ceases to improve on the validation set (i.e., before overfitting begins to occur). (MachineLearningMastery, 2020) 
-	Learning Rate Scheduler – to vary the learning rate for a Deep Learning model, we use the Learning Rate Scheduler call-back that seeks to adjust the learning rate according to a pre-defined schedule. The learning rate function that we tested with was exponential decay.
-	Model Checkpoint – a call-back where a “snapshot” of the state of the system is taken in case of the system failing. If there is a problem in training, the model can revert to this “snapshot”. This is a precautionary call-back.

![image](https://user-images.githubusercontent.com/47773746/128126620-d0357c78-2729-4f9a-9ce3-c81d4436d5d7.png)

### Part 1.4: Feature Extraction for Deep Learning Models

Extracting our features for our Deep Learning models was significantly easier than for our standard models. This is because all we had to do was convert the given images into pixel arrays using the Python library Pydicom, and reformat the array in a way that was acceptable for our Neural Networks (this shape was 128x128x3).

### Part 1.5: Feature Extraction for Standard Models

As mentioned previously, to extract features for our standard models, we had to follow a traditional image pre-processing approach. This entailed extracting specific image features including:
-	Area of opacity
-	The perimeter of visible lung regions shown
-	The various Hu moments
-	The mean and standard deviation of the unenhanced dicom image
-	The various contours in each image, etc

Once we had pre-processed each image, we had then saved the features into a new csv file, which we had loaded up into another file to run our models on.

![image](https://user-images.githubusercontent.com/47773746/128126834-5bcb2c3f-30b5-4fcd-aa3d-3e2f273f0aaa.png)


### Part 1.6: Evaluation Metrics

We utilised the following evaluation metrics for evaluating the performance of our various models:
- Accuracy: this is the quintessential metric regarding classification. It refers to a ratio between the correctly predicted observations over the total number of observations. 
  - A high accuracy on test data is needed to be able to effectively classify radiographs on whether they have Pneumonia or not.
-	Recall (Sensitivity): the ratio of correct positive predictions made from all positive predictions that could have been made.
-	Precision: the ratio of correctly predicted positive observations over the total predicted positive observations.
-	Area under Curve (AUC): provides an aggregate measure of performance across all possible classification thresholds.

We also utilised a Confusion Matrix that allowed us to also effectively summarise the performance of our model’s ability to classify.


## License

[MIT](https://choosealicense.com/licenses/mit/)
