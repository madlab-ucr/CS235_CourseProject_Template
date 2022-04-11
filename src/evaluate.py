'''
Author: Rutuja Gurav

These are some useful evaluation metrics for binary classification. 
Add your task-dependent evaluation metrics here.
Ex - If you are doing a project on unsupervised methods (e.g. clustering), add your evaluation metrics (like elbow plot, NMI, ARI, silhouette coefficient plots, etc) here.

'''

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt
import pandas as pd
import os
# from bcutils4r.eval_model import eval_model # A simple wrapper around scikit-learn and scikit-plot fucntions for evaluating binary classification. Source- https://test.pypi.org/project/bcutils4r/0.0.11/


def generate_classification_report(y_test=None, y_pred=None, class_names=[], save_filepath=None):
    if(len(class_names) !=0):
        report = classification_report(y_test, 
                                        y_pred, 
                                        target_names=[cl+'(class '+str(i)+')' for i, cl in enumerate(class_names)], 
                                        output_dict=True)
    else:
        report = classification_report(y_test, y_pred, output_dict=True)
    
    report_df = pd.DataFrame(report).transpose()
    
    if save_filepath:
        report_df.to_csv(save_filepath)

    return report_df

def plot_roc_curve(y_test=[], y_pred_probas=[], save_filepath=None):
    plot = skplt.metrics.plot_roc(y_test, y_pred_probas)
    if save_filepath:
        p = plot.get_figure()
        p.savefig(save_filepath)
    return plot

def plot_confusion_matrix(y_test=[], y_pred=[], class_names=[], save_filepath=None):
    plot = skplt.metrics.plot_confusion_matrix(y_test, y_pred, labels=class_names, normalize=True)
    if save_filepath:
        p = plot.get_figure()
        p.savefig(save_filepath)
    return plot
