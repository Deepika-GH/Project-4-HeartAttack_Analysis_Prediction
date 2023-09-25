# Heart Attack Prediction Modelling

![header image](https://th.bing.com/th/id/OIG.eGvxvsFQ.HmPQV6DOoy5?pid=ImgGn)

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

This project serves a dual purpose: firstly, it aims to construct a proficient machine learning model capable of forecasting an individual's susceptibility to heart attacks. Secondly, it seeks to design a user-friendly interface that empowers users to self-assess their risk of experiencing a heart attack. This multifaceted analysis encompasses intricate data preprocessing tasks such as managing categorical variables and feature scaling, followed by the training and evaluation of a deep learning model, specifically a neural network, for these predictive tasks. The ultimate objective is to craft a predictive model that can effectively discern individuals who are at heightened risk of heart attacks, delivering substantial value to both the individuals themselves and their families.

## Requirements:


## Data Source:

[Kaggle: Heart Attack Analysis Prediction Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset?resource=download&page=2)

Dataset Description
![image](https://github.com/Deepika-GH/Project-4-HeartAttack_Analysis_Prediction/blob/main/Visuals/Dataset%20Description.png)

| Variable | Description                                     | Values                                                |
|----------|-------------------------------------------------|-------------------------------------------------------|
| age      | patient age                                     | - 1 = male                                           |
| sex      | patient sex                                     | - 0 = female                                         |
| cp       | chest pain type                                | - 0: typical angina                                  |
|          |                                                 | - 1: atypical angina                                 |
|          |                                                 | - 2: non-anginal pain                               |
|          |                                                 | - 3: asymptomatic                                   |
| trtbps   | resting blood pressure (in mm Hg)              | - 1 = true                                           |
| chol     | cholesterol in mg/dl                           | - 0 = normal                                         |
| rest_ecg | resting electrocardiographic results            | - 0 = normal                                         |
|          |                                                 | - 1 = having ST-T wave abnormality                  |
|          |                                                 | - 2 = showing probable or definite left ventricular |
|          |                                                 |     hypertrophy by Estes' criteria                |
|          |                                                 | - 3 = reversable defect                             |
| Variable | Description                                     | Values                                                |
|----------|-------------------------------------------------|-------------------------------------------------------|
| thalach  | maximum heart rate achieved                    | - 1 = yes                                            |
| old peak | ST depression induced by exercise relative to  | - 0 = no                                             |
|          | rest                                            |                                                     |
| slp      | slope of the peak exercise ST segment          | - 0 = unsloping                                      |
|          |                                                 | - 1 = flat                                          |
|          |                                                 | - 2 = downsloping                                   |
| caa      | number of major vessels (0-3)                  | - 0: < 50% diameter narrowing. less chance of       |
|          |                                                 |      heart disease                                  |
|          |                                                 | - 1: > 50% diameter narrowing. more chance of       |
|          |                                                 |      heart disease                                  |
|          |                                                 | - 2 = normal                                        |
| thall    | thalassemia                                    | - 0 = null                                          |
|          |                                                 | - 1 = fixed defect                                  |
|          |                                                 | - 2 = normal                                        |
|          |                                                 | - 3 = reversable defect                             |
| output   | diagnosis of heart disease (angiographic        | - 0: < 50% diameter narrowing. less chance of       |
|          | disease status)                                |      heart disease                                  |
|          |                                                 | - 1: > 50% diameter narrowing. more chance of       |
|          |                                                 |      heart disease                                  |



## Technologies and Libraries:


## Workflow Diagram:

# Modelling

## Principal Component Analysis
### Ron Briggs

## Logical Regression
### Dhawanpreet Dhaliwal

## Random Forest Model
### Leif Munroe

## SVC Linear Testing
### Huma Alam

## Neural Network Model
### Deepika Pitchikala


## Test Case

## Results and Visualizations

## Future Considerations:

## Conclusion:




