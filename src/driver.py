'''
Author: Rutuja Gurav
'''

#Importing required modules
import matplotlib.pyplot as plt 
import numpy as np
from utils import plot_decision_boundary
from dataloader import load_data, split_dataset, transform_data
from models import LogisticRegressionClassifierFromScratch
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model
import os


# Step 1: Load the dataset
data, labels, class_names  = load_data()
print(data.shape, labels.shape, class_names)

# Step 2: Split the dataset into training and testing sets
train_data, train_labels, test_data, test_labels, _, _ = split_dataset(data, labels, test_ratio=0.2)

# Step 3a: Train a logistic regression model from sklearn
print("Logistic Regression model from scikit-learn as a baseline...")
model = LogisticRegression().fit(train_data, train_labels)

# Step 3b: Predict labels for test data
pred_labels = model.predict(test_data)
evaluate_model(true_labels=test_labels, pred_labels=pred_labels, class_names=class_names)

##___________________________STOP HERE FOR MIDTERM REPORT___________________________________________

# Step 4a: Train my logistic regression model
print("My Logistic Regression")
model = LogisticRegressionClassifierFromScratch(l1_penalty=0.0, epochs=100)
model_params, losses = model.train(train_data, train_labels)

# Step 4b: Predict labels for test data
pred_labels = model.predict(test_data, model_params)
evaluate_model(true_labels=test_labels, pred_labels=pred_labels, class_names=class_names)

# # For testing
# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1)
# train_X, train_y, test_X, test_y,_,_ = split_dataset(X, y, test_ratio=0.2)
# model = LogisticRegressionClassifierFromScratch(l1_penalty=1e-6)
# w, b, losses = model.train(train_X, train_y)
# pred_y = model.predict(test_X, w, b)

# evaluate_model(true_labels=test_y, pred_labels=pred_y, class_names=['0','1'])



