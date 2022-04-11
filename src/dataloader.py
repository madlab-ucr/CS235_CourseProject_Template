'''
Author: Rutuja Gurav

Note: This is just an example dataloader, you should customize this for your project.
If you are doing a traditional ML project (not deep learning), you dataloader will look quite similar to this one.
If you are doing a deep learning project you'll likely need data generators to generate batches of data.
'''
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import os

def transform_data(X, option='PCA', n_components=2):
    pca = PCA(n_components)
    X = pca.fit_transform(X)

    return X

def z_normalize_features(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data
    

def load_data():
    DATA_PATH = os.getcwd()+"/data/HTRU_2.csv"
    df = pd.read_csv(DATA_PATH)

    data= df.drop(columns=['target_class']).values
    
    '''
    Z-normalize each feature
    '''
    data = z_normalize_features(data)
    
    '''
    Encode the labels to be numerical
    '''
    le = LabelEncoder()
    labels = le.fit_transform(df.target_class.values)
    class_names = le.classes_

    '''
    Dimensionality reduction of features
    '''
    # data = transform_data(data)

    return data, labels, class_names

def split_dataset(data, labels, train_ratio=0, validation_ratio=0, test_ratio=0):

    if train_ratio and validation_ratio:  
        # Split in 3 -> train, validation and test  
        train_data, leftover_data, \
            train_labels, leftover_labels = train_test_split(data, labels, 
                                                test_size=1 - train_ratio, 
                                                shuffle=True, 
                                                random_state=42)
        val_data, test_data, \
            val_labels, test_labels = train_test_split(leftover_data, 
                                            test_size=test_ratio/(test_ratio + validation_ratio), 
                                            shuffle=True, 
                                            random_state=42) 

    else:
        # Split in 2 -> train and test 
        train_data, test_data, \
            train_labels, test_labels = train_test_split(data, labels, 
                                                test_size=test_ratio, 
                                                shuffle=True, 
                                                random_state=42)
        val_data = None
        val_labels = None

        return train_data, train_labels, test_data, test_labels, val_data, val_labels

    