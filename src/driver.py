'''
Author: Rutuja Gurav

Entry-point script for the project

'''

#Importing required modules
import matplotlib.pyplot as plt 
import numpy as np
from utils import plot_decision_boundary
from dataloader import get_dataset
from models import LogisticRegressionClassifierFromScratch
from sklearn.linear_model import LogisticRegression
from evaluate import generate_classification_report, plot_confusion_matrix, plot_roc_curve
import os

# # Step 1: Get the dataset
train_data, train_labels, test_data, test_labels, class_names = get_dataset(num_splits=2)

# Step 2a: Train a logistic regression model from sklearn
print("Logistic Regression model from scikit-learn as a baseline...")
model = LogisticRegression().fit(train_data, train_labels)

# Step 2b: Predict labels for test data
pred_labels = model.predict(test_data)
pred_probas = model.predict_proba(test_data)
# print(pred_probas)
# Step 2c: Evaluate predictions
report = generate_classification_report(y_test=test_labels, y_pred=pred_labels, \
                                class_names=class_names, \
                                save_filepath=os.getcwd()+"/results/sklearn_cr.csv")
print(report)

plot = plot_confusion_matrix(y_test=test_labels, y_pred=pred_labels, \
                                class_names=[0,1], \
                                save_filepath=os.getcwd()+"/results/sklearn_cm.png")

plot = plot_roc_curve(y_test=test_labels, y_pred_probas=pred_probas, \
                                save_filepath=os.getcwd()+"/results/sklearn_ROCcurve.png")

##___________________________STOP HERE FOR MIDTERM REPORT___________________________________________

# Step 3a: Train my logistic regression model
print("My Logistic Regression")
model = LogisticRegressionClassifierFromScratch(l1_penalty=0.0, epochs=100)
model_params, losses = model.train(train_data, train_labels)

# Step 3b: Predict labels for test data
pred_labels, pred_probas = model.predict(test_data, model_params)
# print(pred_probas)
# Step 3c: Evaluate predictions
report = generate_classification_report(y_test=test_labels, y_pred=pred_labels, \
                                class_names=class_names, \
                                save_filepath=os.getcwd()+"/results/mymodel_cr.csv")
print(report)

plot = plot_confusion_matrix(y_test=test_labels, y_pred=pred_labels, \
                                class_names=[0,1], \
                                save_filepath=os.getcwd()+"/results/mymodel_cm.png")

plot = plot_roc_curve(y_test=test_labels, y_pred_probas=pred_probas, \
                                save_filepath=os.getcwd()+"/results/mymodel_ROCcurve.png")

##___________________________FOR TESTING____________________________________________________________

# from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=100, n_features=10, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1)
# train_X, train_y, test_X, test_y,_,_ = split_dataset(X, y, test_ratio=0.2)

# print("Logistic Regression model from scikit-learn as a baseline...")
# model = LogisticRegression().fit(train_X, train_y)
# pred_y = model.predict(test_X)
# pred_probas = model.predict_proba(test_X)

# report = generate_classification_report(y_test=test_y, y_pred=pred_y, \
#                                             class_names=['0','1'], \
#                                             save_filepath=os.getcwd()+"/results/mymodel_cr.csv"
#                                         )
# print(report)

# print("My Logistic Regression")
# model = LogisticRegressionClassifierFromScratch(l1_penalty=0.0)
# model_params, losses = model.train(train_X, train_y)
# pred_y, pred_probas = model.predict(test_X, model_params)

# report = generate_classification_report(y_test=test_y, y_pred=pred_y, \
#                                             class_names=['0','1'], \
#                                             save_filepath=os.getcwd()+"/results/mymodel_cr.csv"
#                                         )
# print(report)




