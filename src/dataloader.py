'''
Author: Rutuja Gurav

Note: This is just an example dataloader, you should customize this for your project.
If you are doing a traditional ML project (not deep learning), you dataloader will look quite similar to this one.
If you are doing a deep learning project you'll likely need data generators to generate batches of data.
'''

import random
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

import os

def do_pca(X, option='PCA', n_components=2):
    pca = PCA(n_components)
    X = pca.fit_transform(X)

    return X

def z_normalize_features(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data
    

def load_data():

    # Example dataset used in this whole demo
    DATA_PATH = os.getcwd()+"/data/HTRU_2.csv"
    df = pd.read_csv(DATA_PATH)

    data = df.drop(columns=['target_class']).values
    
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
    # data = do_pca(data)

    indices = list(range(len(data)))
    # print(indices)

    return data, labels, class_names, indices

def split_dataset(data, labels, indices, train_ratio=0, validation_ratio=0, test_ratio=0, random_state=42):

    if train_ratio and validation_ratio:  
        # Split in 3 -> train, validation and test  
        train_data, leftover_data, \
            train_labels, leftover_labels,\
                train_indices, leftover_indices = train_test_split(data, labels, indices, 
                                                test_size=1 - train_ratio, 
                                                shuffle=True, 
                                                random_state=random_state)
        val_data, test_data, \
            val_labels, test_labels,\
                val_indices, test_indices = train_test_split(leftover_data, leftover_labels, leftover_indices,
                                            test_size=test_ratio/(test_ratio + validation_ratio), 
                                            shuffle=True, 
                                            random_state=random_state) 

    else:
        # Split in 2 -> train and test 
        train_data, test_data, \
            train_labels, test_labels,\
                train_indices, test_indices = train_test_split(data, labels, indices, 
                                                test_size=test_ratio, 
                                                shuffle=True, 
                                                random_state=random_state)
        val_data = None
        val_labels = None
        val_indices = None

        return train_data, train_labels, train_indices, \
                test_data, test_labels, test_indices, \
                val_data, val_labels, val_indices

def get_dataset(num_splits=2, random_state=42):

    # Step 1: Load the dataset
    data, labels, class_names, indices  = load_data()
    print(data.shape, labels.shape, class_names, len(indices))

    # Step 2: Split the dataset
    if num_splits == 2:
        train_data, train_labels, train_indices, \
            test_data, test_labels, test_indices, \
                _, _, _ = split_dataset(data, labels, indices, 
                                            test_ratio=0.3, 
                                            random_state=random_state
                                        )

        return  train_data, train_labels, train_indices, test_data, test_labels, test_indices, class_names

    elif num_splits == 3:
        train_data, train_labels, train_indices, \
            test_data, test_labels, test_indices, \
                val_data, val_labels, val_indices = split_dataset(data, labels, indices,
                                                        train_ratio = 0.8, 
                                                        validation_ratio = 0.1, 
                                                        test_ratio=0.1, 
                                                        random_state=random_state
                                                    )
        return train_data, train_labels, train_indices, \
                    test_data, test_labels, test_indices, \
                    val_data, val_labels, val_indices, \
                    class_names