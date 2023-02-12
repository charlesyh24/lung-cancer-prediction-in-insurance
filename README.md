# Lung Cancer Prediction in Insurance

## Description
It analyzes lung cancer patient dataset to predict probability of lung cancer development with ML model, and further applies model to insurance planning with three main recommendations.

## Purpose
It is mainly for insurance sales and agents to provide customization in insurance planning for lung cancer with each client with different attributes.

## Usage
After running [main script](code/main.py), you will see several questions about your personal information to answer. While you finish them, insurance recommendations for lung cancer will show up.

## Procedure
1. Preprocessing
    - Data Cleaning
    - Standardization
    - (Dimensionality Reduction)
1. Modeling
    - Logistic Regression
    - Support Vector Machine
    - Random Forest
    - K-Nearest Neighbors
    - Hyperparameter Tuning
1. Application
    - 26~50%: Basic Plan
    - 51~75%: Standard Plan
    - 76+%: Premium Plan

## Content
- data
    - [lung-cancer-patient.csv](data/lung-cancer-patient.csv): lung cancer patient dataset from [here](https://archive.ics.uci.edu/)
- code
    - [eda.ipynb](code/eda.ipynb): EDA report
    - [main.py](code/main.py): main script for running code
    - [preprocessing.py](code/preprocessing.py): sub script for preprocessing
    - [modeling.py](code/modeling.py): sub script for modeling
    - [application.py](code/application.py): sub script for application