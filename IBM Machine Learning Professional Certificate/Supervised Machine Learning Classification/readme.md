# IBM Machine Learning Professional Certificate - Supervised Machine Learning Classification Project

This GitHub project contains the code for the Supervised Machine Learning Classification course of the IBM Machine Learning Professional Certificate. The project is focused on testing and developing a machine learning models to detect credit card fraud.

## Context
Credit card companies need to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. In this project, the dataset contains transactions made by credit cards in September 2013 by European cardholders. This famous dataset from Kaggle presents transactions that occurred in two days, that include 492 frauds out of 284,807 transactions.

## Goal
The goal of the project is to build a bi-classification model based on the Logistic Regression, KNN and SVM to detect fraudulent credit card transactions. This is a particularly challenging task since it involves highly imbalanced classes.

## Project Steps

1. Data Loading: The credit card fraud dataset is loaded using pandas library.

1. Data Exploration and Visualization: Exploratory data analysis (EDA) is performed to gain insights into the dataset. The data is visualized using seaborn and matplotlib libraries.

2. Data Preprocessing: The data is preprocessed by scaling the numerical features using MinMaxScaler/StandardScaler and splitting the dataset into training and testing sets.

3. Model Selection and Training: Logistic Regression, KNN and SVMs are used as our models and they are optimized with minmaxed or standardized data inputs.

4. Model Evaluation: The models are evaluated on the testing data using metrics such as accuracy, precision, recall, and F1-score which are adapted for imbalanced classes.

## Technologies and Packages Used
- Python 3
- Jupyter Notebook
- pandas
- seaborn
- matplotlib
- scikit-learn

The project uses scikit-learn, pandas, seaborn, and matplotlib libraries for machine learning modeling and evaluation, and data manipulation and visualization. The code is written in Python 3 and executed in Jupyter Notebook.
