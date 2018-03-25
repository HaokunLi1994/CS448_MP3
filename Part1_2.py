#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 09:53:27 2018

@author: Haokun Li
"""

import utils
from naive_bayes import NaiveBayes
from bayes_window import Window

if __name__ == '__main__':
    # Global variables
    TRAIN_PATH = ('C:/Users/Shiratori/Desktop/' + 
                  'CS 440 - Artificial Intelligence/' +
                  'mp3/digitdata/optdigits-orig_train.txt')
    TEST_PATH = ('C:/Users/Shiratori/Desktop/' + 
                 'CS 440 - Artificial Intelligence/' +
                 'mp3/digitdata/optdigits-orig_test.txt')
    
    # Load data sets
    X_train, y_train = utils.load_data(TRAIN_PATH)
    X_test, y_test = utils.load_data(TEST_PATH)
    
    # Initialize model
    window = Window(2, 2, overlap=False)
    model = NaiveBayes(laplace=0.1, window=window)
    
    # Fit model
    model.fit(X_train, y_train)
    
    # Predict
    pred_matrix, pred = model.predict(X_test)
    
    # Output
    acc = utils.accuracy(y_test, pred)
    print('Smoothing: {0}'.format(model.laplace))
    print('Accuracy is {0:.3f}\n'.format(acc))
    print('Confusion matrix:')
    print(utils.confusion_matrix(y_test, pred, model))