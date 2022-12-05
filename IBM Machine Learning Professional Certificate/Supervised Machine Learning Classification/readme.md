# [IBM Supervised Machine Learning: Classification](https://www.coursera.org/learn/supervised-machine-learning-classification?specialization=ibm-machine-learning)(Final Project for the course)

## Context

It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. In this project, the dataset contains transactions made by credit cards in September 2013 by European cardholders. This famous [dataset from Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), presents transactions that occurred in two days, that include 492 frauds out of 284,807 transactions.

## Goal

We will try and build a bi-classification model based on the SVM algorithm. This case is of particular interest since it is a bi-classification problem of highly imbalanced classes.

## Note 

There are a few techniques that can be employed in order to deal extreme imbalanced classes like over/under sampling, SMOTE and all sorts of combinations therein. Employing sampling techniques is out of the scope of this particular project.
Our intent is a proof of concept that we can have a decent SVM bi-classification model for imbalanced classes without using explicit sampling techniques.