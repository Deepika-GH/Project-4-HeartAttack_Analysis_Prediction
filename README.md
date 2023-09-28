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
  - [Neural Network Model](#neural-network-model)
- [Test Case](#test-case)
- [Results and Visualizations](#Results-and-Visualizations)
- [Difficulties](#difficulties)
- [Future Considerations](#Future-Considerations)
- [Conclusion](#conclusion)

# Overview

## Introduction:

This project serves a dual purpose: firstly, it aims to construct five different machine learning models capable of forecasting an individual's susceptibility to heart attacks and then choose the best model. Secondly, it seeks to design a user-friendly interface that empowers users to four unique test cases. 

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
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Visuals/png%20plots%20and%20images/Dataset%20Description.png?raw=true)

## Technologies and Libraries:
- pandas, pathlib, matplotlib, numpy, flask, sqlite3, 
- sklearn
  - KMeans, PCA, StandardScalar, train_test_split, RandomForestClassifier
  - balanced_accuracy_score, confusion_matrix, classification_report, make_blobs
- tensorflow, keras_tuner
- pickle, joblib

## Workflow Diagram:
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Visuals/png%20plots%20and%20images/Workflow%20Diagram.png?raw=true)

# Modelling

## Principal Component Analysis
### Ron Briggs

PCA is a dimensionality reduction technique used to transform high-dimensional data into a lower-dimensional representation while preserving the most important information in the data.

**Process**
1. Normalize the data
2. Find the Best Value for k Using the Original Data | Plot Elbow Curve  
3. Cluster with K-means Using the Original Data (n_clusters=2) | Create Scatterplot
4. Optimize Clusters with Principal Component Analysis | Calculate the total explained variance
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Visuals/PCA%20visuals/Variance.png?raw=true)
5. Find the Best Value for k Using the PCA Data | Plot Elbow Curve
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Visuals/PCA%20visuals/ElbowCurvePCA.png?raw=true)
6. Cluster Data with K-means Using the PCA Data | Create Scatterplot
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Visuals/PCA%20visuals/ClustersPCA.png?raw=true)

## Logical Regression
### Dhawanpreet Dhaliwal

## Random Forest Model
### Leif Munroe

## SVC Linear Testing
### Huma Alam

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
. We employed the Keras Tuner library to optimize hyperparameters for our model. The following hyperparameters were tuned:
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

  
## Test Case

## Results and Visualizations

## Future Considerations:

## Conclusion:




