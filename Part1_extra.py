#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 20:48:12 2018

@author: Haokun Li
"""

import time
import utils
from naive_bayes import NaiveBayes
from bayes_window import Window
import numpy as np
import pandas as pd

def load_data():
    """ Load training data and test data
    
    Args:
        path(str): path to folder
    Returns:
        X_train
        y_train
        X_test
        y_test
    """
    X_train_path = 'facedata/facedatatrain'
    y_train_path = 'facedata/facedatatrainlabels'
    X_test_path = 'facedata/facedatatest'
    y_test_path = 'facedata/facedatatestlabels'
    
    # X_train
    f = open(X_train_path)
    X_train = []
    pic = []
    count = 0
    NEW = False
    for line in f:
        if NEW == True:
            NEW = False
            X_train.append(pic.copy())
            pic = []
            count = 0
            
        if count >= 69:
            NEW = True
            
        row = list(line.strip('\n').replace('#', '1').replace(' ', '0'))
        row = [int(x) for x in row]
        pic.append(row)
        
        count += 1
    X_train.append(pic)
    X_train = np.array(X_train)
    f.close()
    
    # y_train
    f = open(y_train_path)
    y_train = []
    for line in f:
        mystr = line.strip()
        y_train.append(int(mystr))
    y_train = np.array(y_train)
    f.close()
    
    # X_test
    f = open(X_test_path)
    X_test = []
    pic = []
    count = 0
    NEW = False
    for line in f:
        if NEW == True:
            NEW = False
            X_test.append(pic.copy())
            pic = []
            count = 0
            
        if count >= 69:
            NEW = True
            
        row = list(line.strip('\n').replace('#', '1').replace(' ', '0'))
        row = [int(x) for x in row]
        pic.append(row)
        
        count += 1
    X_test.append(pic)
    X_test = np.array(X_test)
    f.close()
    
    # y_test
    f = open(y_test_path)
    y_test = []
    for line in f:
        mystr = line.strip()
        y_test.append(int(mystr))
    y_test = np.array(y_test)
    f.close()
    
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    start = time.time()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Initialize model
    window = Window(2, 2, overlap=False)
    model = NaiveBayes(laplace=0.1, window=window)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    pred_matrix, pred = model.predict(X_test)
    
    # Output
    end = time.time()
    acc = utils.accuracy(y_test, pred)
    print('Time used: {0:.3f} second(s)'.format(end - start))
    print('Smoothing: {0}'.format(model.laplace))
    print('Accuracy is {0:.3f}\n'.format(acc))
    print('Confusion matrix:')
    print(pd.DataFrame(utils.confusion_matrix(y_test, pred, model)))