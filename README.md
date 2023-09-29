# Heart Attack Prediction Modelling

![header image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/heart_title_header.jpg?raw=true)

**Authors: Deepika Pitchikala, Leif Munroe, Dhawanpreet Dhaliwal, Huma Alam, Ron Briggs**

## Table of Contents

- [Overview](#overview)
  - [Introduction](#introduction)
  - [Requirements](#requirements)
  - [Data Source](#data-source)
  - [Technologies and Libraries](#technologies-and-libraries)
  - [Workflow Diagram](#workflow-diagram)
- [Modelling](#modelling)
  - [Principal Component Analysis](#principal-component-analysis)
  - [SVC Linear Testing](#SVC-linear-testing)
  - [Random Forest Model](#random-forest-model)
  - [Logistic Regression](#Logistic-Regression)
    - [Classification Report](#Classification-Report)
  - [Neural Network Model](#neural-network-model)
- [Test Cases](#test-cases)
  - [True Positive](#True-Positive)
  - [False Positive](#False-Positive)
  - [True Negative](#True-Negative)
  - [False Negative](#False-Negative)
- [Results and Visualizations](#Results-and-Visualizations)
- [Future Considerations](#Future-Considerations)
- [Conclusions](#conclusions)

# Overview

## Introduction:

This project serves a dual purpose: firstly, it aims to construct five different machine learning models capable of forecasting an individual's susceptibility to heart attacks and then choose the best model. Secondly, it seeks to design a user-friendly interface that empowers users to four test cases based on the confusion matrix. 

This multifaceted analysis encompasses intricate data preprocessing tasks such as managing categorical variables and feature scaling, followed by the training and evaluation of a deep learning model for these predictive tasks. The ultimate objective is to craft a predictive model that can effectively discern individuals who are at heightened risk of heart attacks, delivering substantial value to both the individuals themselves and their families.

## Requirements:
- Use at least 2 of the following: Pandas ✅, Python✅, Matplotlib, SQL ✅, HTML/ CSS/ Bootstrap, Plotly, Leaflet, Tableau, MongoDB, Google Cloud SQL, AmazonAWS
- Use ScikitLearn and or another machine learning library ex: TensorFlow ✅
- The model demonstrates meaningful predictive power at least 75% classification accuracy ✅ or 0.80 R-squared (R squared/ metrics does not apply to our data)
- Optimize predictive power results ✅
- Minimum 100 rows of Data  - our dataset has 300 records ✅
- The model utilizes data retrieved from SQL or Spark - used SQLite ✅

## Data Source:

[Kaggle: Heart Attack Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download&page=2)

Dataset Description
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Dataset%20Description.png?raw=true)

## Technologies and Libraries:
- pandas, pathlib, matplotlib, numpy, flask, sqlite3, 
- sklearn
  - KMeans, PCA, StandardScalar, train_test_split, RandomForestClassifier
  - balanced_accuracy_score, confusion_matrix, classification_report, make_blobs
- tensorflow, keras_tuner
- pickle, joblib

## Workflow Diagram:
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Workflow%20Diagram.png?raw=true)

# Modelling

## Principal Component Analysis
### Ron Briggs

PCA is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional representation while preserving the most important information in the data.

**Process**
1. Normalize the data
2. Find the Best Value for k Using the Original Data | Plot Elbow Curve  
3. Cluster with K-means Using the Original Data (n_clusters=2) | Create Scatterplot
4. Optimize Clusters with Principal Component Analysis | Calculate the total explained variance
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/PCA%20visuals/Variance.png?raw=true)
5. Find the Best Value for k Using the PCA Data | Plot Elbow Curve
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/PCA%20visuals/ElbowCurvePCA.png?raw=true)
6. Cluster Data with K-means Using the PCA Data | Create Scatterplot
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/PCA%20visuals/ClustersPCA.png?raw=true)

## Logical Regression
### Dhawanpreet Dhaliwal

The logistic regression is a statistical method used for analyzing a dataset in which there are one or more independent variables that determine an outcome. It is a type of regression analysis, but it is particularly well-suited for binary classification tasks,

Data Splitting: split the dataset into features (X) and the target variable (y). The features are stored in the X DataFrame, while the target variable is stored in the y Series.

Train-Test Split:Used scikit-learn's train_test_split function to split the data into training and testing sets. This helps in evaluating the model's performance. set random_state=1 for reproducibility.

<img width="631" alt="image" src="https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/assets/130263833/51f7922a-f0b1-4d57-9569-8dcd5551225c">


## Classification Report


<img width="338" alt="image" src="https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/assets/130263833/e660f4b9-f34f-4995-9743-1f6360c7cc72">


Accuracy:  This mode got 86% of them right. This means the model is doing a pretty good job overall in identifying whether someone has heart disease or not.


Precision: This is like how careful we are when predicting someone has heart disease. It tells us that out of all the predictions we made for "no heart disease," 88% of them were really cases of "no heart disease." So, This model is quite accurate when it predicts no heart disease.


F1-Score: It's like a balance between precision and recall. The F1-Score is 84, which is a good balance. This means we're doing a decent job at both being accurate when predicting no heart disease and not missing too many cases of heart disease.


In summary, this model is doing a pretty good job in predicting heart disease, especially for cases where there is no heart disease. We're capturing most of the actual cases, but there's room for improvement to ensure we don't miss any. 



## Random Forest Model
### Leif Munroe
A Random Forest Model was chosen for its flexibility and tested for it's accuracy and F1 scores when predicting the likelyhood that an indivdual is at risk of experiencing a heart attack. After the data was split into testing and training set, it was scaled to provide a more optimized outcome. After this the model was fitted and the predictions made. 

The following image of the classification report and confusion matrix show the analysis of the prediciton results. The overall accuracy is 80%, but the f1 scores of 78% for the not-at-risk group and 82% for the at-risk group show the result discrepancy that leads to lower model performance. Additionally, with support numbers of 39 (0-not at risk) and 37 (1-at risk) we can see that the data is fairly well balanced. 

![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/assets/126816323/792a9bf5-6553-4251-a328-c2367a429f1e)


## SVC Linear Testing
### Huma Alam
GOAL: Build a SVC Linear model and test its accuracy/ recall to see which machine learning model helps best predict the outcome of an individual suffering from a heart attack or likelihood of not suffering from an heart attack.
I created a SVC model, with our trained X (features) and y (target):
 
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/assets/130116747/9783c760-2979-49f1-b0d2-3aed42a35c66)

The model outputted 88% precision of predicting No Heart Disease and 86% for predicting Heart Disease. The f1-score has a combination of precision and recall which helps us see 0.85 score of predicting No Heart Disease and 0.88 for Heart Disease. With scoring the highest f-1 score, we have choosen this model to help trained our machine learning model.

![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/assets/130116747/fd464341-b235-489c-b9fd-cf29d09ce47f)

Confusion Matrix:


![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/assets/130116747/5e91c35a-3b89-4583-ac67-471672483e91)



## Neural Network Model
### Deepika Pitchikala
- Model Overview
. Built a neural network model for binary classification (predicting whether a patient is prone to getting a heart attck or no). The model architecture is as follows:
  
. Input Layer: The input layer has 13 neurons, representing the 13 input features.

. Hidden Layers: The number of hidden layers and neurons in each hidden layer is determined using hyperparameter tuning. We allow the Keras Tuner to decide the activation function and the number of neurons in each hidden layer.

. Output Layer: The output layer consists of a single neuron with a sigmoid activation function, which produces the probability of having heart disease.

. Loss Function: We used binary cross-entropy as the loss function, suitable for binary classification tasks.

. Optimizer: We used the Adam optimizer to train the model.
  
- Hyperparameter Tuning
  
 We employed the Keras Tuner library to optimize hyperparameters for our model. The following hyperparameters were tuned:

. Activation function for hidden layers (choices: relu, tanh, sigmoid)

. Number of neurons in the first layer (range: 1 to 64, step: 2)

. Number of hidden layers (range: 1 to 6)

. Number of neurons in each hidden layer (range: 1 to 64, step: 2)

- Best Hyperparameters: The best hyperparameters were determined using random search.

- Model Performance:
  
. Accuracy on the test dataset: [0.6184]

. Loss on the test dataset: [0.6856]

. Classification Report on the test dataset:

![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/assets/77449446/10405ec8-b022-4287-b5c3-3770dffa469b)

## Test Cases

### True Positive
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/true_positive_case.png?raw=true)

### False Positive
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/false_positive_case.png?raw=true)

### True Negative
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/true_negative_case.png?raw=true)

### False Negative
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/false_negative_case.png?raw=true)
    
## Results and Visualizations

![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/ModelAccuracyComparison.png?raw=true)

![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/Distribution_plot.png?raw=true)

![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/binary_countplot.png?raw=true)

![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/static/corr_matrix.png?raw=true)

## Future Considerations:
In the future, it would be prudent to grade the predicting models using recall rather than accuracy. This would be necessary to reduce the error from false negatives. Although, using the F1 scores ultimately proved to be an effective way to choose the best predictive model.

## Conclusions:
- PCA required 9 components to have a total explained variance of over 80% when there are 12 features in the model data
- SVC Linear model scored the highest of all models in terms of F1 score and accuracy
- True Positives have the highest prediction rate while False Negatives have the lowest prediction rate




